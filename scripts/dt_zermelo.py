'''
Trains Decision Transformer on an offline dataset of the Zermelo navigation environment.

Decision Transformer treats offline RL as sequence modeling: given a desired
return-to-go, past states, and past actions, it predicts the next action.
At eval time, we condition on a high target return to get goal-reaching behavior.

=== FULL WORKFLOW ===
# (Identical to bc_zermelo.py except step 5 trains DT instead of BC.)
# See bc_zermelo.py docstring for the full sequence.

CUDA_VISIBLE_DEVICES=7 python ~/zermelo-navigation/scripts/dt_zermelo.py \
    --train_steps=500000 \
    --seed=0 \
    --proj_wandb=zermelo_hit_dynamic_poordataset \
    --run_group=dt \
    --wandb_online=True
'''
import argparse
import json
import os
import random
import sys
import time

os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

# Make scripts/ importable for shared helpers, and repo root for `import zermelo_env`.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))

from helpers import training_common as tc  # noqa: E402
from zermelo_env.zermelo_config import load_config  # noqa: E402


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DecisionTransformer(nn.Module):
    """GPT-style Decision Transformer for continuous control.

    Input sequence per timestep: [return-to-go, state, action]
    The model predicts actions given (rtg, state) context.
    """

    def __init__(self, obs_dim, act_dim, hidden_dim=128, n_heads=4,
                 n_layers=3, context_len=100, max_ep_len=1024, dropout=0.1):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.context_len = context_len

        self.embed_rtg = nn.Linear(1, hidden_dim)
        self.embed_state = nn.Linear(obs_dim, hidden_dim)
        self.embed_action = nn.Linear(act_dim, hidden_dim)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_dim)
        self.embed_ln = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.predict_action = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh(),
        )

    def forward(self, rtgs, states, actions, timesteps):
        """
        Args:
            rtgs:      (B, T, 1)
            states:    (B, T, obs_dim)
            actions:   (B, T, act_dim)
            timesteps: (B, T) int
        Returns:
            action_preds: (B, T, act_dim)
        """
        B, T = states.shape[0], states.shape[1]

        rtg_emb = self.embed_rtg(rtgs)
        state_emb = self.embed_state(states)
        action_emb = self.embed_action(actions)
        time_emb = self.embed_timestep(timesteps)
        rtg_emb = rtg_emb + time_emb
        state_emb = state_emb + time_emb
        action_emb = action_emb + time_emb

        # Interleave [rtg_0, state_0, action_0, rtg_1, ...] → (B, 3*T, H).
        stacked = torch.stack([rtg_emb, state_emb, action_emb], dim=2)
        stacked = stacked.reshape(B, 3 * T, self.hidden_dim)
        stacked = self.embed_ln(stacked)

        seq_len = 3 * T
        if not hasattr(self, '_causal_mask') or self._causal_mask.shape[0] != seq_len:
            self._causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=stacked.device), diagonal=1).bool()
        hidden = self.transformer(stacked, mask=self._causal_mask)

        # Extract state token positions (indices 1, 4, 7, ...).
        state_hidden = hidden[:, 1::3, :]
        return self.predict_action(state_hidden)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DTDataset:
    """Samples random sub-sequences from episodes for DT training.

    Pre-pads all episodes to context_len and concatenates into contiguous
    arrays so sampling is a single vectorised gather (no Python for-loop).
    """

    def __init__(self, episodes, context_len, obs_mean, obs_std,
                 max_ep_len=1024, rtg_scale=100.0, device='cpu'):
        self.context_len = context_len
        self.max_ep_len = max_ep_len
        self.device = device
        K = context_len

        obs_dim = episodes[0]['observations'].shape[1]
        act_dim = episodes[0]['actions'].shape[1]

        # Build a flat table of all valid (episode, start) windows.
        # For each window we store a pre-normalised, zero-padded chunk.
        all_states, all_actions, all_rtgs, all_timesteps, all_masks = [], [], [], [], []
        for ep in episodes:
            obs = (ep['observations'] - obs_mean) / obs_std
            acts = ep['actions']
            rtg = ep['rtg'] / rtg_scale
            ep_len = len(obs)
            n_windows = max(1, ep_len - K + 1)
            for start in range(n_windows):
                length = min(K, ep_len - start)
                s = np.zeros((K, obs_dim), dtype=np.float32)
                a = np.zeros((K, act_dim), dtype=np.float32)
                r = np.zeros((K, 1), dtype=np.float32)
                t = np.zeros(K, dtype=np.int64)
                m = np.zeros(K, dtype=np.float32)
                s[:length] = obs[start:start + length]
                a[:length] = acts[start:start + length]
                r[:length, 0] = rtg[start:start + length]
                t[:length] = np.arange(start, start + length).clip(0, max_ep_len - 1)
                m[:length] = 1.0
                all_states.append(s)
                all_actions.append(a)
                all_rtgs.append(r)
                all_timesteps.append(t)
                all_masks.append(m)

        self.states = np.stack(all_states)       # (N, K, obs_dim)
        self.actions = np.stack(all_actions)      # (N, K, act_dim)
        self.rtgs = np.stack(all_rtgs)            # (N, K, 1)
        self.timesteps = np.stack(all_timesteps)  # (N, K)
        self.masks = np.stack(all_masks)          # (N, K)
        self.n_windows = len(all_states)
        print(f"  DTDataset: {self.n_windows:,} windows pre-computed")

    def sample(self, batch_size):
        idx = np.random.randint(0, self.n_windows, size=batch_size)
        return {
            'states': torch.as_tensor(self.states[idx], device=self.device),
            'actions': torch.as_tensor(self.actions[idx], device=self.device),
            'rtgs': torch.as_tensor(self.rtgs[idx], device=self.device),
            'timesteps': torch.as_tensor(self.timesteps[idx], device=self.device),
            'masks': torch.as_tensor(self.masks[idx], device=self.device),
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_dt(model, env, num_episodes, target_return, obs_mean, obs_std,
                context_len, max_ep_len, rtg_scale, device):
    """Evaluate DT by conditioning on target_return."""
    obs_dim = len(obs_mean)
    act_dim = env.action_space.shape[0]
    returns, lengths, successes = [], [], []

    for _ in range(num_episodes):
        ob, _ = env.reset()
        done = False
        ep_ret, ep_len = 0.0, 0

        states_buf = np.zeros((context_len, obs_dim), dtype=np.float32)
        actions_buf = np.zeros((context_len, act_dim), dtype=np.float32)
        rtgs_buf = np.zeros((context_len, 1), dtype=np.float32)
        timesteps_buf = np.zeros(context_len, dtype=np.int64)
        remaining_return = target_return

        while not done:
            t = min(ep_len, context_len - 1)
            if ep_len >= context_len:
                states_buf[:-1] = states_buf[1:]
                actions_buf[:-1] = actions_buf[1:]
                rtgs_buf[:-1] = rtgs_buf[1:]
                timesteps_buf[:-1] = timesteps_buf[1:]
                t = context_len - 1

            states_buf[t] = (ob - obs_mean) / obs_std
            rtgs_buf[t, 0] = remaining_return / rtg_scale
            timesteps_buf[t] = min(ep_len, max_ep_len - 1)

            seq_len = t + 1
            s = torch.tensor(states_buf[:seq_len], device=device).unsqueeze(0)
            a = torch.tensor(actions_buf[:seq_len], device=device).unsqueeze(0)
            r = torch.tensor(rtgs_buf[:seq_len], device=device).unsqueeze(0)
            ts = torch.tensor(timesteps_buf[:seq_len], device=device).unsqueeze(0)

            with torch.no_grad():
                action_preds = model(r, s, a, ts)
            action = np.clip(action_preds[0, -1].cpu().numpy(), -1.0, 1.0)
            actions_buf[t] = action

            ob, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_ret += reward
            remaining_return -= reward
            ep_len += 1

        returns.append(ep_ret)
        lengths.append(ep_len)
        successes.append(info.get('success', 0.0))

    return {
        'episode.return': np.mean(returns),
        'episode.length': np.mean(lengths),
        'success': np.mean(successes),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Decision Transformer on Zermelo')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_steps', type=int, default=500000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--context_len', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_ep_len', type=int, default=1024)
    parser.add_argument('--rtg_scale', type=float, default=100.0)
    parser.add_argument('--target_return', type=float, default=None,
                        help='Target return for eval. Defaults to max return in dataset.')
    parser.add_argument('--eval_interval', type=int, default=50000)
    parser.add_argument('--eval_episodes', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=50000)
    parser.add_argument('--log_interval', type=int, default=5000)
    parser.add_argument('--zermelo_dataset', type=str, default=None)
    parser.add_argument('--zermelo_config', type=str, default=None)
    parser.add_argument('--proj_wandb', type=str, default=None)
    parser.add_argument('--run_group', type=str, default='dt')
    parser.add_argument('--wandb_entity', type=str, default='RL_Control_JX')
    parser.add_argument('--wandb_online', type=bool, default=True)
    parser.add_argument('--save_dir', type=str, default='exp/')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"Device: {device}")

    zermelo_cfg = load_config(args.zermelo_config)
    zermelo_cfg_src = tc.default_config_src_path(args.zermelo_config)
    args.proj_wandb = args.proj_wandb or zermelo_cfg['wandb_project_name']

    # Wandb.
    exp_name = f'dt_sd{args.seed:03d}_{time.strftime("%Y%m%d_%H%M%S")}'
    entity = args.wandb_entity if args.wandb_entity != 'None' else None
    os.environ["WANDB_MODE"] = "online" if args.wandb_online else "offline"
    wandb.init(project=args.proj_wandb, group=args.run_group, name=exp_name,
               entity=entity,
               config={**vars(args), 'zermelo_config_yaml': zermelo_cfg})
    if os.path.isfile(zermelo_cfg_src):
        wandb.save(zermelo_cfg_src, policy='now')

    save_dir = os.path.join(args.save_dir, args.proj_wandb, args.run_group, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'flags.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(save_dir, 'zermelo_config.json'), 'w') as f:
        json.dump(zermelo_cfg, f, indent=2)

    # Dataset.
    dataset_path = args.zermelo_dataset or tc.default_dataset_path(zermelo_cfg)
    data, train_segs, val_segs = tc.load_episode_segments(dataset_path)
    train_episodes = tc.episode_views(data, train_segs, with_rtg=True)
    val_episodes = tc.episode_views(data, val_segs, with_rtg=True)
    all_returns = [ep['rewards'].sum() for ep in train_episodes + val_episodes]
    print(f"  Return range: [{min(all_returns):.1f}, {max(all_returns):.1f}], "
          f"mean={np.mean(all_returns):.1f}")

    # Observation normalization (compute from train episodes).
    all_train_obs = np.concatenate([ep['observations'] for ep in train_episodes])
    obs_mean, obs_std = tc.compute_obs_norm(all_train_obs)
    tc.save_obs_norm(save_dir, obs_mean, obs_std)

    train_ds = DTDataset(train_episodes, args.context_len, obs_mean, obs_std,
                         max_ep_len=args.max_ep_len, rtg_scale=args.rtg_scale, device=device)
    val_ds = DTDataset(val_episodes, args.context_len, obs_mean, obs_std,
                       max_ep_len=args.max_ep_len, rtg_scale=args.rtg_scale, device=device)

    if args.target_return is None:
        args.target_return = max(all_returns)
        print(f"  Auto target_return: {args.target_return:.1f} (max in dataset)")

    obs_dim = train_episodes[0]['observations'].shape[1]
    act_dim = train_episodes[0]['actions'].shape[1]
    model = DecisionTransformer(
        obs_dim=obs_dim, act_dim=act_dim, hidden_dim=args.hidden_dim,
        n_heads=args.n_heads, n_layers=args.n_layers,
        context_len=args.context_len, max_ep_len=args.max_ep_len,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    print(f"DT params: {sum(p.numel() for p in model.parameters()):,}")

    eval_env = tc.make_eval_env(args.zermelo_config)

    # Compile model for faster execution (PyTorch 2.0+).
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
        print("  torch.compile enabled")

    # AMP (mixed precision) for faster matmuls.
    scaler = torch.amp.GradScaler('cuda')
    amp_dtype = torch.float16

    # Training.
    first_time = time.time()
    last_time = time.time()
    for step in tqdm(range(1, args.train_steps + 1), desc="DT Training", dynamic_ncols=True):
        batch = train_ds.sample(args.batch_size)
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            action_preds = model(batch['rtgs'], batch['states'], batch['actions'], batch['timesteps'])
            mask = batch['masks'].unsqueeze(-1)
            loss = (((action_preds - batch['actions']) ** 2) * mask).sum() / mask.sum() / act_dim

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        scaler.step(optimizer)
        scaler.update()

        if step % args.log_interval == 0:
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=amp_dtype):
                val_batch = val_ds.sample(args.batch_size)
                val_preds = model(val_batch['rtgs'], val_batch['states'],
                                  val_batch['actions'], val_batch['timesteps'])
                val_mask = val_batch['masks'].unsqueeze(-1)
                val_loss = (((val_preds - val_batch['actions']) ** 2) * val_mask).sum() / val_mask.sum() / act_dim
            wandb.log({
                'training/loss': loss.item(),
                'validation/loss': val_loss.item(),
                'time/epoch_time': (time.time() - last_time) / args.log_interval,
                'time/total_time': time.time() - first_time,
            }, step=step)
            last_time = time.time()

        if step % args.eval_interval == 0:
            model.eval()
            eval_info = evaluate_dt(
                model, eval_env, args.eval_episodes, args.target_return,
                obs_mean, obs_std, args.context_len, args.max_ep_len,
                args.rtg_scale, device,
            )
            model.train()
            wandb.log({f'evaluation/{k}': v for k, v in eval_info.items()}, step=step)
            print(f"  Step {step}: return={eval_info['episode.return']:.2f}, "
                  f"success={eval_info['success']:.2f}, "
                  f"length={eval_info['episode.length']:.0f}")

        if step % args.save_interval == 0:
            # Strip '_orig_mod.' prefix added by torch.compile so checkpoints
            # load cleanly into an uncompiled model.
            sd = {k.removeprefix('_orig_mod.'): v for k, v in model.state_dict().items()}
            torch.save(sd, os.path.join(save_dir, f'model_{step}.pt'))

    wandb.finish()
    print("Done.")


if __name__ == '__main__':
    main()
