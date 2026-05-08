#!/usr/bin/env python3
"""
Unified evaluation script for all trained PointMaze agents.

This script orchestrates evaluation across two conda environments:
  - 'flowrl' for MeanFlowQL (needs JAX/Flax)
  - 'pointmaze' for SB3 baselines (needs matching numpy version)

It does this by spawning subprocesses that run the actual eval code
in the correct conda environment. The results are saved as JSON files
and merged into a final avg_returns.txt.

Usage (run from any environment):
    CUDA_VISIBLE_DEVICES=0 python ~/evaluate_all.py

Output:
    /ehome/niti/Results/
        avg_returns.txt
        eval_videos/
            MeanFlowQL/   (3 videos)
            DDPG/          (3 videos)
            TD3/           (3 videos)
            SAC/           (3 videos)
            PPO_config_1/  (3 videos)
            PPO_config_2/  (3 videos)
"""

import os
import sys
import json
import subprocess
import numpy as np

RESULTS_DIR = '/ehome/niti/Results'
VIDEO_DIR = os.path.join(RESULTS_DIR, 'eval_videos')
RETURNS_FILE = os.path.join(RESULTS_DIR, 'avg_returns.txt')
MEANFLOWQL_RESULTS_FILE = os.path.join(RESULTS_DIR, 'meanflowql_results.json')
SB3_RESULTS_FILE = os.path.join(RESULTS_DIR, 'sb3_results.json')

# Conda environment python paths
FLOWRL_PYTHON = os.path.expanduser('~/miniconda3/envs/flowrl/bin/python')
POINTMAZE_PYTHON = os.path.expanduser('~/miniconda3/envs/pointmaze/bin/python')

# The two worker scripts
MEANFLOWQL_SCRIPT = os.path.expanduser('~/evaluate_meanflowql.py')
SB3_SCRIPT = os.path.expanduser('~/evaluate_sb3.py')

NUM_EVAL_EPISODES = 1000
MAX_EPISODE_STEPS = 300


def run_in_env(python_path, script_path, label):
    """Run a script using a specific conda env's python interpreter."""
    print(f'\n{"=" * 60}')
    print(f'Running {label}...')
    print(f'  Python: {python_path}')
    print(f'  Script: {script_path}')
    print(f'{"=" * 60}\n')

    result = subprocess.run(
        [python_path, script_path],
        env={
            **os.environ,
            'PYOPENGL_PLATFORM': 'egl',
            'MUJOCO_GL': 'egl',
            'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
            'NUM_EVAL_EPISODES': str(NUM_EVAL_EPISODES),
            'MAX_EPISODE_STEPS': str(MAX_EPISODE_STEPS),
        },
    )

    if result.returncode != 0:
        print(f'\nERROR: {label} failed with exit code {result.returncode}')
        return False
    return True


def merge_results():
    """Merge JSON results from both eval scripts into avg_returns.txt."""
    all_results = {}

    for path, label in [
        (MEANFLOWQL_RESULTS_FILE, 'MeanFlowQL'),
        (SB3_RESULTS_FILE, 'SB3 baselines'),
    ]:
        try:
            with open(path, 'r') as f:
                all_results.update(json.load(f))
            print(f'Loaded {label} results from {path}')
        except FileNotFoundError:
            print(f'WARNING: {path} not found ({label} eval may have failed)')

    if not all_results:
        print('No results to merge.')
        return

    print('\n' + '=' * 60)
    print('RESULTS SUMMARY')
    print('=' * 60)

    with open(RETURNS_FILE, 'w') as f:
        f.write('PointMaze UMaze-v3 Evaluation Results (continuing_task=True)\n')
        f.write(f'Episodes per algorithm: {NUM_EVAL_EPISODES}\n')
        f.write(f'Max steps per episode: {MAX_EPISODE_STEPS}\n')
        f.write(f'{"=" * 60}\n\n')

        sorted_results = sorted(
            all_results.items(),
            key=lambda x: x[1]['avg_return'],
            reverse=True,
        )
        for algo_name, data in sorted_results:
            returns = data['all_returns']
            avg = data['avg_return']
            std = float(np.std(returns))
            mn = float(np.min(returns))
            mx = float(np.max(returns))
            line = f'{algo_name:20s}  avg={avg:7.3f}  std={std:6.3f}  min={mn:7.3f}  max={mx:7.3f}'
            print(line)
            f.write(line + '\n')

    print(f'\nResults written to: {RETURNS_FILE}')
    print(f'Videos saved to:    {VIDEO_DIR}')


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)

    # Step 1: Evaluate MeanFlowQL in flowrl env
    run_in_env(FLOWRL_PYTHON, MEANFLOWQL_SCRIPT, 'MeanFlowQL evaluation (flowrl env)')

    # Step 2: Evaluate SB3 baselines in pointmaze env
    run_in_env(POINTMAZE_PYTHON, SB3_SCRIPT, 'SB3 baseline evaluation (pointmaze env)')

    # Step 3: Merge results
    merge_results()


if __name__ == '__main__':
    main()
