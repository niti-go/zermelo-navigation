"""Load and merge Zermelo config from YAML files.

Provides a single dict with all env + dataset params.  Both the environment
(ZermeloMazeEnv) and the generation script read through this module so there
is one source of truth.
"""

import copy
import os

import yaml


# Path to the default config shipped with the repo.
_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'configs', 'zermelo_config.yaml',
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


def config_to_env_kwargs(cfg):
    """Extract the kwargs that ZermeloMazeEnv.__init__ expects from a config dict."""
    maze_map = cfg['maze']['map'] if cfg['maze']['enabled'] else None

    # Dynamic flow sub-config (may be absent in older YAML files).
    dynamic_flow_cfg = cfg['flow'].get('dynamic', {})

    return dict(
        maze_type=cfg['maze']['type'],
        maze_unit=cfg['maze']['unit'],
        maze_height=cfg['maze']['height'],
        maze_on=cfg['maze']['enabled'],
        maze_map_override=maze_map,
        flow_field_path=cfg['flow']['field_path'],
        dynamic_flow_cfg=dynamic_flow_cfg,
        fixed_start_goal=cfg['start_goal']['fixed'],
        fixed_init_ij=tuple(cfg['start_goal']['start_ij']),
        fixed_goal_ij=tuple(cfg['start_goal']['goal_ij']),
        goal_reward=cfg['reward']['goal_reward'],
        energy_weight=cfg['reward']['energy_weight'],
        time_weight=cfg['reward']['time_weight'],
        distance_weight=cfg['reward']['distance_weight'],
        drift_threshold=cfg['reward']['drift_threshold'],
        ob_type=cfg['observation']['type'],
        include_flow_in_obs=cfg['observation']['include_flow'],
        terminate_at_goal=cfg['env']['terminate_at_goal'],
        goal_tolerance=cfg['env']['goal_tolerance'],
        add_noise_to_goal=cfg['env']['add_noise_to_goal'],
        success_timing=cfg['env']['success_timing'],
    )
