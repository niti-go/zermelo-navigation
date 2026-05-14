"""Load and merge Zermelo config from YAML files.

Provides a single dict with all env + dataset params.  Both the environment
(ZermeloMazeEnv) and the generation script read through this module so there
is one source of truth.
"""

import os

import yaml


# Path to the default config shipped with the repo.
_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'zermelo_config.yaml',
)


def _deep_merge(base, overrides):
    """Recursively merge *overrides* into *base* (mutates base)."""
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(config_path=None):
    """Return the full Zermelo config dict.

    Args:
        config_path: Optional path to a user YAML file.  Values in this file
            override the defaults.  If None, only the built-in defaults are
            used.

    Returns:
        A plain dict mirroring the YAML structure.
    """
    with open(_DEFAULT_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    if config_path is not None:
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f) or {}
        _deep_merge(cfg, user_cfg)

    return cfg


def build_hit_flow_cfg(cfg, max_file=None):
    """Translate the YAML `flow` block into kwargs for HITChainFlow.

    `max_file` overrides `flow.max_file` (use this in dataset generators to
    restrict to the train segment, e.g. max_file=45). The env itself does
    NOT apply a max_file by default; eval / replay code can query any
    available frame.
    """
    flow_cfg = cfg['flow']
    nc_dir = flow_cfg['nc_dir']
    if not os.path.isabs(nc_dir):
        # Resolve relative paths against the repo root.
        repo_root = os.path.dirname(_DEFAULT_CONFIG_PATH)
        nc_dir = os.path.abspath(os.path.join(repo_root, nc_dir))

    out = {
        'nc_dir': nc_dir,
        'x_range': flow_cfg.get('x_range', [-4.0, 24.0]),
        'y_range': flow_cfg.get('y_range', [-4.0, 24.0]),
        'n_tiles': flow_cfg.get('n_tiles', 1.0),
        'target_max': flow_cfg.get('target_max', None),
        'frames_per_step': float(flow_cfg.get('frames_per_step', 1.0)),
    }
    chosen_max = max_file if max_file is not None else flow_cfg.get('max_file', None)
    if chosen_max is not None:
        out['max_file'] = int(chosen_max)
    return out


def config_to_env_kwargs(cfg, max_file=None):
    """Extract the kwargs that ZermeloMazeEnv.__init__ expects from a config dict."""
    maze_map = cfg['maze']['map'] if cfg['maze']['enabled'] else None

    hit_flow_cfg = build_hit_flow_cfg(cfg, max_file=max_file)

    task_cfg = cfg['task']
    reward_cfg = cfg['reward']

    energy_cfg = reward_cfg['energy']

    return dict(
        maze_unit=cfg['maze']['unit'],
        maze_height=cfg['maze']['height'],
        maze_on=cfg['maze']['enabled'],
        maze_map_override=maze_map,
        hit_flow_cfg=hit_flow_cfg,
        fixed_start_goal=(task_cfg['start_goal_mode'] == 'fixed'),
        fixed_init_ij=tuple(task_cfg['fixed_start_ij']),
        fixed_goal_ij=tuple(task_cfg['fixed_goal_ij']),
        goal_reward=reward_cfg['goal_reward'],
        action_weight=energy_cfg['action_weight'],
        fixed_hover_cost=energy_cfg['fixed_hover_cost'],
        progress_weight=reward_cfg['progress_weight'],
        drift_threshold=reward_cfg['drift_threshold'],
        ob_type=cfg['observation']['type'],
        include_flow_in_obs=cfg['observation']['include_flow'],
        terminate_at_goal=cfg['env']['terminate_at_goal'],
        goal_tolerance=cfg['env']['goal_tolerance'],
        action_scale=cfg['env'].get('action_scale', 2.0),
        add_noise_to_goal=task_cfg['noise_on_reset'],
        add_noise_to_start=task_cfg['noise_on_reset'],
        show_action_arrow=cfg['env'].get('show_action_arrow', False),
    )
