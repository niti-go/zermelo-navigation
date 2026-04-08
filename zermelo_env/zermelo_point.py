import os

import mujoco
import numpy as np

from zermelo_env.point import PointEnv
from zermelo_env.zermelo_flow import DynamicTGVFlowField, FlowField


class ZermeloPointEnv(PointEnv):
    """Point mass environment with a background fluid flow field (Zermelo navigation).

    The flow field adds a displacement to the agent's position at each step,
    simulating navigation through a fluid. The field can be either:
      - Static: loaded from an .npy/.npz file (default).
      - Dynamic: an analytically computed, time-varying Taylor-Green vortex
        controlled via ``dynamic_flow_cfg``.
    """

    def __init__(self, flow_field_path=None, include_flow_in_obs=True,
                 dynamic_flow_cfg=None, **kwargs):
        self._flow_field_path = flow_field_path
        self._include_flow_in_obs = include_flow_in_obs

        # Decide between static file-based flow and dynamic analytic flow.
        self._dynamic_flow_cfg = dynamic_flow_cfg or {}
        self._use_dynamic_flow = self._dynamic_flow_cfg.get('enabled', False)

        if self._use_dynamic_flow:
            self._flow_field = DynamicTGVFlowField(self._dynamic_flow_cfg)
        else:
            self._flow_field = FlowField(flow_field_path)

        # Simulation time — reset to 0 each episode.
        self._sim_time = 0.0

        super().__init__(**kwargs)
        # Note: observation_space is NOT updated here. The parent PointEnv sets a
        # placeholder (6,) for MuJoCo init, and the maze wrapper (ZermeloMazeEnv)
        # overrides it to match get_ob()'s actual output shape after construction.
        # This mirrors how the original MazeEnv works with PointEnv.

    def step(self, action):
        prev_qpos = self.data.qpos.copy()
        prev_qvel = self.data.qvel.copy()

        action = 0.2 * action

        # Agent's intended displacement
        self.data.qpos[:] = self.data.qpos + action

        # Flow displacement: dt * flow_velocity
        dt = self.frame_skip * self.model.opt.timestep
        x, y = self.data.qpos[0], self.data.qpos[1]
        if self._use_dynamic_flow:
            flow_vx, flow_vy = self._flow_field.get_flow(x, y, self._sim_time)
        else:
            flow_vx, flow_vy = self._flow_field.get_flow(x, y)
        self.data.qpos[0] += dt * flow_vx
        self.data.qpos[1] += dt * flow_vy

        self.data.qvel[:] = np.array([0.0, 0.0])

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        # Advance simulation clock.
        self._sim_time += dt

        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        observation = self.get_ob()

        if self.render_mode == 'human':
            self.render()

        return (
            observation,
            0.0,
            False,
            False,
            {
                'xy': self.get_xy(),
                'prev_qpos': prev_qpos,
                'prev_qvel': prev_qvel,
                'qpos': qpos,
                'qvel': qvel,
            },
        )

    def reset_model(self):
        self._sim_time = 0.0
        return super().reset_model()

    def get_ob(self):
        base_ob = self.data.qpos.flat.copy()
        if self._include_flow_in_obs:
            x, y = self.data.qpos[0], self.data.qpos[1]
            if self._use_dynamic_flow:
                flow_vx, flow_vy = self._flow_field.get_flow(x, y, self._sim_time)
            else:
                flow_vx, flow_vy = self._flow_field.get_flow(x, y)
            return np.concatenate([base_ob, [flow_vx, flow_vy]])
        return base_ob
