"""Test script for the Zermelo point maze environment.

Runs a random policy, prints observations and flow info, and saves a video.

Usage:
    python scripts/test_zermelo.py
    python scripts/test_zermelo.py --save_video oracle_test.mp4
    python scripts/test_zermelo.py --policy oracle --flow_field_path ogbench/locomaze/assets/my_field.npy
    python scripts/test_zermelo.py --live  # Watch live in a matplotlib window
"""
import argparse
import os

import gymnasium
import numpy as np

import zermelo_env  # noqa

VIDEO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'videos')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='zermelo-pointmaze-medium-v0')
    parser.add_argument('--num_steps', type=int, default=200)
    parser.add_argument('--save_video', type=str, default=None, help='Filename (saved in ogbench/videos/)')
    parser.add_argument('--policy', type=str, default='random', choices=['random', 'oracle', 'zero'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--flow_field_path', type=str, default=None, help='Path to a .npy flow field file')
    parser.add_argument('--live', action='store_true', help='Show live rendering in a matplotlib window')
    # Reward shaping parameters.
    parser.add_argument('--goal_reward', type=float, default=1.0, help='Reward for reaching goal')
    parser.add_argument('--action_weight', type=float, default=0.0,
                        help='Energy cost per unit ||action|| (dynamic component)')
    parser.add_argument('--fixed_hover_cost', type=float, default=0.0,
                        help='Per-step baseline energy to stay airborne in still air')
    parser.add_argument('--progress_weight', type=float, default=0.0, help='Reward weight for distance reduction')
    parser.add_argument('--timeout_penalty', type=float, default=0.0, help='Penalty when episode times out')
    args = parser.parse_args()

    np.random.seed(args.seed)

    env_kwargs = dict(
        terminate_at_goal=False,
        max_episode_steps=args.num_steps + 10,
        goal_reward=args.goal_reward,
        action_weight=args.action_weight,
        fixed_hover_cost=args.fixed_hover_cost,
        progress_weight=args.progress_weight,
        timeout_penalty=args.timeout_penalty,
    )
    if args.flow_field_path is not None:
        env_kwargs['flow_field_path'] = args.flow_field_path

    env = gymnasium.make(args.env_name, **env_kwargs)

    ob, info = env.reset(options=dict(task_id=1))
    print(f'Environment: {args.env_name}')
    if args.flow_field_path:
        print(f'Flow field: {args.flow_field_path}')
    print(f'Observation space: {env.observation_space}')
    print(f'Action space: {env.action_space}')
    print(f'Initial observation: {ob}')
    print(f'Goal observation shape: {info["goal"].shape}')
    print(f'Agent position: {env.unwrapped.get_xy()}')
    print(f'Goal position: {env.unwrapped.cur_goal_xy}')
    print()

    # Set up live display if requested.
    fig, ax, img_plot = None, None, None
    if args.live:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.ion()
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.set_axis_off()
        frame = env.unwrapped.render()
        img_plot = ax.imshow(frame)
        fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)

    frames = []
    frame = env.unwrapped.render()
    frames.append(frame)

    for step in range(args.num_steps):
        if args.policy == 'random':
            action = env.action_space.sample()
        elif args.policy == 'zero':
            action = np.zeros(env.action_space.shape)
        elif args.policy == 'oracle':
            subgoal_xy, _ = env.unwrapped.get_oracle_subgoal(
                env.unwrapped.get_xy(), env.unwrapped.cur_goal_xy
            )
            subgoal_dir = subgoal_xy - env.unwrapped.get_xy()
            subgoal_dir = subgoal_dir / (np.linalg.norm(subgoal_dir) + 1e-6)
            action = np.clip(subgoal_dir, -1, 1)

        ob, reward, terminated, truncated, info = env.step(action)
        frame = env.unwrapped.render()
        frames.append(frame)

        if args.live and img_plot is not None:
            img_plot.set_data(frame)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            import matplotlib.pyplot as plt
            plt.pause(0.05)

        if step % 20 == 0 or step == args.num_steps - 1:
            xy = env.unwrapped.get_xy()
            flow = env.unwrapped._flow_field.get_flow(xy[0], xy[1])
            drift_str = 'drift' if info.get('is_drifting', 0) > 0.5 else 'move'
            print(f'Step {step:4d} | pos=({xy[0]:6.2f}, {xy[1]:6.2f}) | '
                  f'flow=({flow[0]:5.2f}, {flow[1]:5.2f}) | '
                  f'd={info.get("dist_to_goal", 0):5.2f} | '
                  f'r={reward:6.3f} | {drift_str} | success={info["success"]:.0f}')

        if terminated or truncated:
            print(f'Episode ended at step {step} (terminated={terminated}, truncated={truncated})')
            break

    if args.save_video:
        os.makedirs(VIDEO_DIR, exist_ok=True)
        video_path = os.path.join(VIDEO_DIR, args.save_video)
        try:
            import imageio
            imageio.mimwrite(video_path, np.array(frames), fps=10)
            print(f'\nVideo saved to {video_path}')
        except ImportError:
            print('\nInstall imageio to save videos: pip install imageio imageio-ffmpeg')
            fallback = video_path.replace('.mp4', '_frames.npy')
            np.save(fallback, np.array(frames))
            print(f'Frames saved to {fallback}')
    else:
        print(f'\nTip: pass --save_video test.mp4 to save a video (goes to {VIDEO_DIR}/).')
        print('Tip: pass --policy zero to see pure flow drift (no agent action).')
        print('Tip: pass --policy oracle to see BFS-guided navigation through flow.')
        print('Tip: pass --live to watch the episode in real time.')
        print('Tip: use --action_weight, --fixed_hover_cost, --progress_weight for reward shaping.')

    if args.live:
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.show()

    env.close()


if __name__ == '__main__':
    main()
