'''
Trains Decision Transformer on an offline dataset of the Zermelo navigation environment.

Decision Transformer treats offline RL as sequence modeling: given a desired
return-to-go, past states, and past actions, it predicts the next action.
At eval time, we condition on a high target return to get goal-reaching behavior.

=== FULL WORKFLOW (run these in order) ===

# 1. Start a tmux session
tmux new -s dt_zermelo

# 2. Generate the offline dataset (uses zermelo conda env)
conda activate zermelo
cd ~/zermelo-navigation
PYTHONPATH=. python scripts/generate_dataset.py

# 3. Find good reward weights using the analysis script
PYTHONPATH=. python scripts/analyze_rewards.py
#    Check the recommended weights in the output and plots in datasets/hyperparameter_tuning/.

# 4. Recompute rewards in the existing dataset (no regeneration needed)
PYTHONPATH=. python scripts/recompute_rewards.py \
    --energy_weight=<recommended ew> \
    --time_weight=<recommended tw> \
    --distance_weight=<recommended dw>

# 5. Visualize trajectories from the dataset (optional)
PYTHONPATH=. python scripts/visualize.py
#    This saves a video of 5 random trajectories to datasets/video.mp4

# 6. Train DT (uses flowrl conda env, can run from any directory)
conda activate flowrl
wandb login
CUDA_VISIBLE_DEVICES=7 python ~/zermelo-navigation/experiments/zermelo/dt_zermelo.py \
    --train_steps=500000 \
    --seed=0 \
    --proj_wandb=zermelo_hit_dynamic_poordataset\
    --run_group=dt \
    --wandb_online=True

# 7. Detach tmux: Ctrl+b, then d
# 8. Reattach later: tmux attach -t dt_zermelo

=== WANDB LOGGING ===
  Entity:  --wandb_entity (default: RL_Control_JX)
  Project: --proj_wandb   (default: zermelo)
  Group:   --run_group    (default: dt)
  Mode:    --wandb_online (default: True, set False for offline logging)

=== CHECKPOINTS ===
  Saved to: exp/<proj_wandb>/<run_group>/<exp_name>/
  Includes: model checkpoints, obs_norm_stats.npz, flags.json
'''
import argparse
import json
import math
import os
import random
import sys
import time

os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['MUJOCO_GL'] = 'egl'

import gymnasium
import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

# Zermelo env imports.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _repo_root)
import zermelo_env  # noqa — registers gymnasium envs
from zermelo_env.zermelo_config import load_config, config_to_env_kwargs


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

        # Token embeddings: project each modality to hidden_dim.
        self.embed_rtg = nn.Linear(1, hidden_dim)
        self.embed_state = nn.Linear(obs_dim, hidden_dim)
        self.embed_action = nn.Linear(act_dim, hidden_dim)

        # Learned positional embedding (one per absolute timestep, shared across token types).
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_dim)

        # LayerNorm applied after embedding.
        self.embed_ln = nn.LayerNorm(hidden_dim)

        # Transformer encoder with causal masking (self-attention only).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Action prediction head (applied to state tokens).
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

        # Embed each modality.
        rtg_emb = self.embed_rtg(rtgs)        # (B, T, H)
        state_emb = self.embed_state(states)    # (B, T, H)
        action_emb = self.embed_action(actions) # (B, T, H)

        # Add timestep embeddings.
        time_emb = self.embed_timestep(timesteps)  # (B, T, H)
        rtg_emb = rtg_emb + time_emb
        state_emb = state_emb + time_emb
        action_emb = action_emb + time_emb

        # Interleave: [rtg_0, state_0, action_0, rtg_1, state_1, action_1, ...]
        # Shape: (B, 3*T, H)
        stacked = torch.stack([rtg_emb, state_emb, action_emb], dim=2)  # (B, T, 3, H)
        stacked = stacked.reshape(B, 3 * T, self.hidden_dim)
        stacked = self.embed_ln(stacked)

        # Causal mask: each token can attend to itself and all previous tokens.
        seq_len = 3 * T
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=stacked.device), diagonal=1).bool()

        # Transformer encoder with causal self-attention.
        hidden = self.transformer(stacked, mask=causal_mask)

        # Extract state token positions (indices 1, 4, 7, ...).
        state_hidden = hidden[:, 1::3, :]  # (B, T, H)

        # Predict actions from state representations.
        action_preds = self.predict_action(state_hidden)  # (B, T, act_dim)
        return action_preds


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_zermelo_episodes(dataset_path):
    """Load Zermelo .npz and return list of episode dicts."""
    print(f"Loading dataset: {dataset_path}")
    data = dict(np.load(dataset_path))

    obs = data['observations'].astype(np.float32)
    actions = data['actions'].astype(np.float32)
    rewards = data['rewards'].astype(np.float32)
    terminals = data['terminals'].astype(np.float32)

    # Split into episodes.
    ends = np.where(terminals > 0.5)[0]
    starts = np.concatenate([[0], ends[:-1] + 1])

    episodes = []
    for s, e in zip(starts, ends + 1):
        ep_rewards = rewards[s:e]
        # Compute returns-to-go (cumulative reward from each step to end of episode).
        rtg = np.zeros_like(ep_rewards)
        rtg[-1] = ep_rewards[-1]
        for t in reversed(range(len(ep_rewards) - 1)):
            rtg[t] = ep_rewards[t] + rtg[t + 1]
        episodes.append({
            'observations': obs[s:e],
            'actions': actions[s:e],
            'rewards': ep_rewards,
            'rtg': rtg,
        })

    print(f"  {len(episodes)} episodes, obs_dim={obs.shape[1]}, act_dim={actions.shape[1]}")
    returns = [ep['rewards'].sum() for ep in episodes]
    print(f"  Return range: [{min(returns):.1f}, {max(returns):.1f}], "
          f"mean={np.mean(returns):.1f}")
    return episodes


class DTDataset:
    """Samples random sub-sequences from episodes for Decision Transformer training."""

    def __init__(self, episodes, context_len, obs_mean, obs_std, max_ep_len=1024,
                 rtg_scale=100.0, device='cpu'):
        self.episodes = episodes
        self.context_len = context_len
        self.max_ep_len = max_ep_len
        self.rtg_scale = rtg_scale
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.device = device
        # Weight sampling by episode length so longer episodes are sampled proportionally.
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

            # Random start position.
            if ep_len <= K:
                start = 0
                length = ep_len
            else:
                start = np.random.randint(0, ep_len - K + 1)
                length = K

            # Normalize observations.
            obs_norm = (ep['observations'][start:start + length] - self.obs_mean) / self.obs_std

            states[i, :length] = torch.tensor(obs_norm, device=self.device)
            actions[i, :length] = torch.tensor(ep['actions'][start:start + length], device=self.device)
            rtgs[i, :length, 0] = torch.tensor(
                ep['rtg'][start:start + length] / self.rtg_scale, device=self.device)
            timesteps[i, :length] = torch.arange(start, start + length, device=self.device)
            masks[i, :length] = 1.0

        # Clamp timesteps to max_ep_len - 1 for the positional embedding.
        timesteps = timesteps.clamp(0, self.max_ep_len - 1)

        return {
            'states': states,
            'actions': actions,
            'rtgs': rtgs,
            'timesteps': timesteps,
            'masks': masks,
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def make_eval_env(zermelo_config_path=None):
    cfg = load_config(zermelo_config_path)
    env_kwargs = config_to_env_kwargs(cfg)
    env_kwargs['fixed_start_goal'] = True
    env_kwargs['max_episode_steps'] = cfg['run']['max_episode_steps']
    return gymnasium.make('zermelo-pointmaze-medium-v0', **env_kwargs)


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

        # Rolling context buffers.
        states_buf = np.zeros((context_len, obs_dim), dtype=np.float32)
        actions_buf = np.zeros((context_len, act_dim), dtype=np.float32)
        rtgs_buf = np.zeros((context_len, 1), dtype=np.float32)
        timesteps_buf = np.zeros(context_len, dtype=np.int64)

        remaining_return = target_return

        while not done:
            t = min(ep_len, context_len - 1)
            # Shift buffers left if we've exceeded context.
            if ep_len >= context_len:
                states_buf[:-1] = states_buf[1:]
                actions_buf[:-1] = actions_buf[1:]
                rtgs_buf[:-1] = rtgs_buf[1:]
                timesteps_buf[:-1] = timesteps_buf[1:]
                t = context_len - 1

            states_buf[t] = (ob - obs_mean) / obs_std
            rtgs_buf[t, 0] = remaining_return / rtg_scale
            timesteps_buf[t] = min(ep_len, max_ep_len - 1)

            # Build input tensors up to current timestep.
            seq_len = t + 1
            s = torch.tensor(states_buf[:seq_len], device=device).unsqueeze(0)
            a = torch.tensor(actions_buf[:seq_len], device=device).unsqueeze(0)
            r = torch.tensor(rtgs_buf[:seq_len], device=device).unsqueeze(0)
            ts = torch.tensor(timesteps_buf[:seq_len], device=device).unsqueeze(0)

            with torch.no_grad():
                action_preds = model(r, s, a, ts)
            action = action_preds[0, -1].cpu().numpy()
            action = np.clip(action, -1.0, 1.0)

            # Store action in buffer.
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
    parser.add_argument('--max_ep_len', type=int, default=1024,
                        help='Max episode length for positional embeddings.')
    parser.add_argument('--rtg_scale', type=float, default=100.0,
                        help='Scale factor to normalize returns-to-go.')
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

    # Seeds.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load the zermelo YAML config up front so we can log it alongside argparse flags.
    zermelo_cfg = load_config(args.zermelo_config)
    zermelo_cfg_src = args.zermelo_config or os.path.join(
        _repo_root, 'zermelo_config.yaml')

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
    if args.zermelo_dataset is None:
        args.zermelo_dataset = os.path.join(_repo_root, zermelo_cfg['run']['save_path'])

    all_episodes = load_zermelo_episodes(args.zermelo_dataset)

    # Train/val split by episode.
    np.random.shuffle(all_episodes)
    n_train = int(len(all_episodes) * 0.8)
    train_episodes = all_episodes[:n_train]
    val_episodes = all_episodes[n_train:]
    print(f"  Train episodes: {n_train}, Val episodes: {len(all_episodes) - n_train}")

    # Observation normalization (compute from train episodes).
    all_train_obs = np.concatenate([ep['observations'] for ep in train_episodes])
    obs_mean = all_train_obs.mean(axis=0)
    obs_std = all_train_obs.std(axis=0) + 1e-6
    np.savez(os.path.join(save_dir, 'obs_norm_stats.npz'), obs_mean=obs_mean, obs_std=obs_std)

    train_ds = DTDataset(train_episodes, args.context_len, obs_mean, obs_std,
                         max_ep_len=args.max_ep_len, rtg_scale=args.rtg_scale, device=device)
    val_ds = DTDataset(val_episodes, args.context_len, obs_mean, obs_std,
                       max_ep_len=args.max_ep_len, rtg_scale=args.rtg_scale, device=device)

    # Target return for evaluation.
    if args.target_return is None:
        args.target_return = max(ep['rewards'].sum() for ep in all_episodes)
        print(f"  Auto target_return: {args.target_return:.1f} (max in dataset)")

    # Model.
    obs_dim = all_episodes[0]['observations'].shape[1]
    act_dim = all_episodes[0]['actions'].shape[1]
    model = DecisionTransformer(
        obs_dim=obs_dim, act_dim=act_dim, hidden_dim=args.hidden_dim,
        n_heads=args.n_heads, n_layers=args.n_layers,
        context_len=args.context_len, max_ep_len=args.max_ep_len,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"DT params: {sum(p.numel() for p in model.parameters()):,}")

    # Eval env.
    eval_env = make_eval_env(args.zermelo_config)

    # Training.
    first_time = time.time()
    last_time = time.time()

    for step in tqdm(range(1, args.train_steps + 1), desc="DT Training", dynamic_ncols=True):
        batch = train_ds.sample(args.batch_size)

        action_preds = model(batch['rtgs'], batch['states'], batch['actions'], batch['timesteps'])

        # MSE loss on action predictions, masked to valid timesteps.
        mask = batch['masks'].unsqueeze(-1)  # (B, T, 1)
        loss = (((action_preds - batch['actions']) ** 2) * mask).sum() / mask.sum() / act_dim

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()

        # Log.
        if step % args.log_interval == 0:
            with torch.no_grad():
                val_batch = val_ds.sample(args.batch_size)
                val_preds = model(val_batch['rtgs'], val_batch['states'],
                                  val_batch['actions'], val_batch['timesteps'])
                val_mask = val_batch['masks'].unsqueeze(-1)
                val_loss = (((val_preds - val_batch['actions']) ** 2) * val_mask).sum() / val_mask.sum() / act_dim

            metrics = {
                'training/loss': loss.item(),
                'validation/loss': val_loss.item(),
                'time/epoch_time': (time.time() - last_time) / args.log_interval,
                'time/total_time': time.time() - first_time,
            }
            wandb.log(metrics, step=step)
            last_time = time.time()

        # Eval.
        if step % args.eval_interval == 0:
            model.eval()
            eval_info = evaluate_dt(
                model, eval_env, args.eval_episodes, args.target_return,
                obs_mean, obs_std, args.context_len, args.max_ep_len,
                args.rtg_scale, device,
            )
            model.train()
            eval_metrics = {f'evaluation/{k}': v for k, v in eval_info.items()}
            wandb.log(eval_metrics, step=step)
            print(f"  Step {step}: return={eval_info['episode.return']:.2f}, "
                  f"success={eval_info['success']:.2f}, length={eval_info['episode.length']:.0f}")

        # Save.
        if step % args.save_interval == 0:
            ckpt_path = os.path.join(save_dir, f'model_{step}.pt')
            torch.save(model.state_dict(), ckpt_path)

    wandb.finish()
    print("Done.")


if __name__ == '__main__':
    main()
