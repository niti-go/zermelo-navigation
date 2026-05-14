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

from utils import training_common as tc  # noqa: E402
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
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=stacked.device), diagonal=1).bool()
        hidden = self.transformer(stacked, mask=causal_mask)

        # Extract state token positions (indices 1, 4, 7, ...).
        state_hidden = hidden[:, 1::3, :]
        return self.predict_action(state_hidden)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DTDataset:
    """Samples random sub-sequences from episodes for DT training."""

    def __init__(self, episodes, context_len, obs_mean, obs_std,
                 max_ep_len=1024, rtg_scale=100.0, device='cpu'):
        self.episodes = episodes
        self.context_len = context_len
        self.max_ep_len = max_ep_len
        self.rtg_scale = rtg_scale
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.device = device
        lens = np.array([len(ep['observations']) for ep in episodes])
        self.weights = lens / lens.sum()

    def sample(self, batch_size):
        obs_dim = self.episodes[0]['observations'].shape[1]
        act_dim = self.episodes[0]['actions'].shape[1]
        K = self.context_len

        states = torch.zeros(batch_size, K, obs_dim, device=self.device)
        actions = torch.zeros(batch_size, K, act_dim, device=self.device)
        rtgs = torch.zeros(batch_size, K, 1, device=self.device)
        timesteps = torch.zeros(batch_size, K, dtype=torch.long, device=self.device)
        masks = torch.zeros(batch_size, K, device=self.device)

        ep_indices = np.random.choice(len(self.episodes), size=batch_size, p=self.weights)
        for i, ep_idx in enumerate(ep_indices):
            ep = self.episodes[ep_idx]
            ep_len = len(ep['observations'])

            if ep_len <= K:
                start, length = 0, ep_len
            else:
                start = np.random.randint(0, ep_len - K + 1)
                length = K

            obs_norm = (ep['observations'][start:start + length] - self.obs_mean) / self.obs_std

            states[i, :length] = torch.tensor(obs_norm, device=self.device)
            actions[i, :length] = torch.tensor(
                ep['actions'][start:start + length], device=self.device)
            rtgs[i, :length, 0] = torch.tensor(
                ep['rtg'][start:start + length] / self.rtg_scale, device=self.device)
            timesteps[i, :length] = torch.arange(start, start + length, device=self.device)
            masks[i, :length] = 1.0

        timesteps = timesteps.clamp(0, self.max_ep_len - 1)
        return {'states': states, 'actions': actions, 'rtgs': rtgs,
                'timesteps': timesteps, 'masks': masks}


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
    parser.add_argument('--batch_size', type=int, default=64)
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
    parser.add_argument('--proj_wandb', type=str, default='zermelo')
    parser.add_argument('--run_group', type=str, default='dt')
    parser.add_argument('--wandb_entity', type=str, default='RL_Control_JX')
    parser.add_argument('--wandb_online', type=bool, default=True)
    parser.add_argument('--save_dir', type=str, default='exp/')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    zermelo_cfg = load_config(args.zermelo_config)
    zermelo_cfg_src = tc.default_config_src_path(args.zermelo_config)

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

    # Training.
    first_time = time.time()
    last_time = time.time()
    for step in tqdm(range(1, args.train_steps + 1), desc="DT Training", dynamic_ncols=True):
        batch = train_ds.sample(args.batch_size)
        action_preds = model(batch['rtgs'], batch['states'], batch['actions'], batch['timesteps'])
        mask = batch['masks'].unsqueeze(-1)
        loss = (((action_preds - batch['actions']) ** 2) * mask).sum() / mask.sum() / act_dim

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()

        if step % args.log_interval == 0:
            with torch.no_grad():
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
            torch.save(model.state_dict(), os.path.join(save_dir, f'model_{step}.pt'))

    wandb.finish()
    print("Done.")


if __name__ == '__main__':
    main()
