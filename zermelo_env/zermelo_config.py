"""Load and merge Zermelo config from YAML files.

Provides a single dict with all env + dataset params.  Both the environment
(ZermeloMazeEnv) and the generation script read through this module so there
is one source of truth.
"""

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


def get_flow_runtime(cfg):
    """Public helper: return (dynamic_flow_cfg, static_flow_path) from a loaded
    config. Other scripts (visualize, evaluate_models) use this instead of
    poking at cfg['flow'] directly so the schema stays in one place.

    `dynamic_flow_cfg` has the legacy {'enabled': bool, 'mode': ..., ...} shape
    that the env / DynamicFlowField classes consume. `static_flow_path` is the
    .npy/.npz path when mode='static', else None.
    """
    flow_cfg = cfg['flow']
    high_ram = bool(cfg.get('system', {}).get('high_ram_memory', False))
    return _build_dynamic_flow_cfg(flow_cfg, high_ram), _static_flow_path(flow_cfg)


def _build_dynamic_flow_cfg(flow_cfg, high_ram_memory=False):
    """Translate the user-facing `flow.mode` switch into the legacy
    dynamic_flow_cfg dict that ZermeloMazeEnv / ZermeloPointEnv consume.

    The env code expects: {'enabled': bool, 'mode': 'tgv'|'netcdf', ...params,
    'x_range': [...], 'y_range': [...]}.
    """
    mode = flow_cfg.get('mode', 'static')
    if mode == 'static':
        return {'enabled': False}

    out = {
        'enabled': True,
        'mode': mode,
        'x_range': flow_cfg.get('x_range', [-4.0, 24.0]),
        'y_range': flow_cfg.get('y_range', [-4.0, 24.0]),
    }
    if mode == 'tgv':
        out.update(flow_cfg.get('tgv', {}))
    elif mode == 'netcdf':
        netcdf_cfg = dict(flow_cfg.get('netcdf', {}))
        # Honor the system-wide high-RAM flag unless the netcdf block sets
        # preload_all_frames explicitly.
        netcdf_cfg.setdefault('preload_all_frames', bool(high_ram_memory))
        out['netcdf'] = netcdf_cfg
    else:
        raise ValueError(f"flow.mode must be one of 'static', 'tgv', 'netcdf'; got {mode!r}")
    return out


def _static_flow_path(flow_cfg):
    """Return the static .npy/.npz path, or None if a dynamic mode is active."""
    if flow_cfg.get('mode', 'static') != 'static':
        return None
    return flow_cfg.get('static', {}).get('path')


def config_to_env_kwargs(cfg):
    """Extract the kwargs that ZermeloMazeEnv.__init__ expects from a config dict."""
    maze_map = cfg['maze']['map'] if cfg['maze']['enabled'] else None

    flow_cfg = cfg['flow']
    high_ram = bool(cfg.get('system', {}).get('high_ram_memory', False))
    dynamic_flow_cfg = _build_dynamic_flow_cfg(flow_cfg, high_ram)
    flow_field_path = _static_flow_path(flow_cfg)

    task_cfg = cfg['task']
    reward_cfg = cfg['reward']

    return dict(
        maze_unit=cfg['maze']['unit'],
        maze_height=cfg['maze']['height'],
        maze_on=cfg['maze']['enabled'],
        maze_map_override=maze_map,
        flow_field_path=flow_field_path,
        dynamic_flow_cfg=dynamic_flow_cfg,
        fixed_start_goal=(task_cfg['start_goal_mode'] == 'fixed'),
        fixed_init_ij=tuple(task_cfg['fixed_start_ij']),
        fixed_goal_ij=tuple(task_cfg['fixed_goal_ij']),
        # logging.goal_reward is the only weight the env actually uses at
        # eval time; the rest are recomputed downstream from raw components.
        goal_reward=reward_cfg['goal_reward'],
        energy_weight=reward_cfg['energy_weight'],
        time_weight=reward_cfg['time_weight'],
        distance_weight=reward_cfg['distance_weight'],
        drift_threshold=reward_cfg['drift_threshold'],
        ob_type=cfg['observation']['type'],
        include_flow_in_obs=cfg['observation']['include_flow'],
        terminate_at_goal=cfg['env']['terminate_at_goal'],
        goal_tolerance=cfg['env']['goal_tolerance'],
        add_noise_to_goal=task_cfg['noise_on_reset'],
        add_noise_to_start=task_cfg['noise_on_reset'],
        show_action_arrow=cfg['env'].get('show_action_arrow', False),
    )
