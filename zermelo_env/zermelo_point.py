import mujoco
import numpy as np

from zermelo_env.point import PointEnv
from zermelo_env.hit_chain import HITChainFlow


class ZermeloPointEnv(PointEnv):
    """Point mass environment with a chained-HIT background flow field.

    The flow advances in *frame* coordinates: each env step advances the
    internal frame counter by ``frames_per_step`` (which may be fractional).
    The flow at any continuous frame index is provided by ``HITChainFlow``.

    Episode reset can override the starting frame via
    ``reset(options={'start_frame': F})``; otherwise the episode begins at
    frame 0.
    """

    def __init__(self, hit_flow_cfg, include_flow_in_obs=True,
                 action_scale=2.0, **kwargs):
        cfg = dict(hit_flow_cfg)
        self._frames_per_step = float(cfg.pop('frames_per_step', 1.0))
        self._flow_field = HITChainFlow(**cfg)
        self._include_flow_in_obs = include_flow_in_obs
        # Per-axis multiplier from raw action ∈ [-1, 1] to agent velocity in
        # world units. With action_scale = 2 the agent's max speed is
        # 2*sqrt(2) ≈ 2.83 w.u./s, comparable to the flow's peak.
        self._action_scale = float(action_scale)

        # Continuous frame index of the *current* env state.
        self._frame = 0.0
        # Frame index at the start of the current episode.
        self._start_frame = 0.0

        super().__init__(**kwargs)

    @property
    def frame(self):
        return self._frame

    @property
    def frames_per_step(self):
        return self._frames_per_step

    @property
    def n_frames(self):
        return self._flow_field.n_frames

    def step(self, action):
        prev_qpos = self.data.qpos.copy()
        prev_qvel = self.data.qvel.copy()

        # Both agent and flow contribute displacement = dt * velocity.
        # Raw action ∈ [-1, 1]² is rescaled by self._action_scale to give an
        # agent velocity comparable to the flow's peak speed.
        dt = self.frame_skip * self.model.opt.timestep
        agent_vx = self._action_scale * action[0]
        agent_vy = self._action_scale * action[1]

        x, y = self.data.qpos[0], self.data.qpos[1]
        flow_vx, flow_vy = self._flow_field.get_flow(x, y, self._frame)

        self.data.qpos[0] += dt * (agent_vx + flow_vx)
        self.data.qpos[1] += dt * (agent_vy + flow_vy)

        self.data.qvel[:] = np.array([0.0, 0.0])

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        # Advance the flow clock.
        self._frame += self._frames_per_step

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
                'frame': float(self._frame),
            },
        )

    def reset_model(self):
        self._frame = float(self._start_frame)
        return super().reset_model()

    def set_start_frame(self, start_frame):
        """Pin the next reset to begin at this flow-clock frame."""
        self._start_frame = float(start_frame)

    def set_frame(self, frame):
        """Override the current flow-clock frame (used by replay/visualize)."""
        self._frame = float(frame)

    def get_ob(self):
        base_ob = self.data.qpos.flat.copy()
        if self._include_flow_in_obs:
            x, y = self.data.qpos[0], self.data.qpos[1]
            flow_vx, flow_vy = self._flow_field.get_flow(x, y, self._frame)
            return np.concatenate([base_ob, [flow_vx, flow_vy]])
        return base_ob
