import math
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from gymnasium.spaces import Box

from zermelo_env.zermelo_flow import FlowField
from zermelo_env.zermelo_point import ZermeloPointEnv


def make_zermelo_maze_env(*args, **kwargs):
    """Factory function for creating a Zermelo maze environment.

    This creates a point maze with a background fluid flow field.
    Only supports point locomotion and the 'maze' environment type.

    Args:
        *args: Additional arguments to pass to ZermeloMazeEnv.
        **kwargs: Additional keyword arguments including flow_field_path.
    """

    class ZermeloMazeEnv(ZermeloPointEnv):
        """Maze environment with Zermelo fluid flow.

        Inherits from ZermeloPointEnv and adds maze walls plus flow visualization arrows.
        """

        def __init__(
            self,
            maze_type='medium',
            maze_unit=4.0,
            maze_height=0.5,
            terminate_at_goal=True,
            success_timing='post',
            ob_type='states',
            add_noise_to_goal=True,
            reward_task_id=None,
            use_oracle_rep=False,
            flow_field_path=None,
            include_flow_in_obs=True,
            fixed_start_goal=False,
            fixed_init_ij=(6, 1),
            fixed_goal_ij=(1, 6),
            maze_on=True,
            maze_map_override=None,
            # Reward parameters.
            goal_reward=1.0,
            energy_weight=0.0,
            time_weight=0.0,
            distance_weight=0.0,
            drift_threshold=0.01,
            goal_tolerance=1.0,
            *args,
            **kwargs,
        ):
            self._maze_type = maze_type
            self._maze_unit = maze_unit
            self._maze_height = maze_height
            self._terminate_at_goal = terminate_at_goal
            self._success_timing = success_timing
            self._ob_type = ob_type
            self._add_noise_to_goal = add_noise_to_goal
            self._reward_task_id = reward_task_id
            self._use_oracle_rep = use_oracle_rep
            self._fixed_start_goal = fixed_start_goal
            self._fixed_init_ij = tuple(fixed_init_ij)
            self._fixed_goal_ij = tuple(fixed_goal_ij)
            self._maze_on = maze_on

            # Reward parameters.
            self._goal_reward = goal_reward
            self._energy_weight = energy_weight
            self._time_weight = time_weight
            self._distance_weight = distance_weight
            self._drift_threshold = drift_threshold

            assert ob_type in ['states', 'pixels']
            assert success_timing in ['pre', 'post']

            # Define constants.
            self._offset_x = 4
            self._offset_y = 4
            self._noise = 1
            self._goal_tol = goal_tolerance

            # Load flow field early for visualization.
            self._flow_for_arrows = FlowField(flow_field_path)

            # --- Build maze map ---
            if maze_map_override is not None:
                # Use the caller-supplied map (from config file).
                maze_map = maze_map_override
            elif self._maze_type == 'medium':
                if self._maze_on:
                    maze_map = [
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 1, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 0, 1],
                        [1, 1, 0, 0, 0, 1, 1, 1],
                        [1, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 1, 0, 0, 1, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                    ]
                else:
                    # Open arena: outer boundary walls only, no internal walls.
                    maze_map = [
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                    ]
            else:
                raise ValueError(f'Zermelo maze only supports medium type, got: {self._maze_type}')

            self.maze_map = np.array(maze_map)

            # Collect free cells for variable start/goal sampling.
            self._free_cells = []
            for i in range(self.maze_map.shape[0]):
                for j in range(self.maze_map.shape[1]):
                    if self.maze_map[i, j] == 0:
                        self._free_cells.append((i, j))

            # Update XML file.
            xml_file = self.xml_file
            tree = ET.parse(xml_file)
            self.update_tree(tree)
            _, maze_xml_file = tempfile.mkstemp(text=True, suffix='.xml')
            tree.write(maze_xml_file)

            super().__init__(
                xml_file=maze_xml_file,
                flow_field_path=flow_field_path,
                include_flow_in_obs=include_flow_in_obs,
                *args,
                **kwargs,
            )

            # Make custom camera.
            if self.camera_id is None and self.camera_name is None:
                camera = mujoco.MjvCamera()
                camera.lookat[0] = 2 * (self.maze_map.shape[1] - 3)
                camera.lookat[1] = 2 * (self.maze_map.shape[0] - 3)
                camera.distance = 5 * (self.maze_map.shape[1] - 2)
                camera.elevation = -90
                self.custom_camera = camera
            else:
                self.custom_camera = self.camera_id or self.camera_name

            # Set task goals.
            self.task_infos = []
            self.cur_task_id = None
            self.cur_task_info = None
            self.set_tasks()
            self.num_tasks = len(self.task_infos)
            self.cur_goal_xy = np.zeros(2)

            self.custom_renderer = None
            if self._ob_type == 'pixels':
                self.observation_space = Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
                tex_grid = self.model.tex('grid')
                tex_height = tex_grid.height[0]
                tex_width = tex_grid.width[0]
                attr_name = 'tex_rgb' if hasattr(self.model, 'tex_rgb') else 'tex_data'
                tex_rgb = getattr(self.model, attr_name)[tex_grid.adr[0]:tex_grid.adr[0] + 3 * tex_height * tex_width]
                tex_rgb = tex_rgb.reshape(tex_height, tex_width, 3)
                for x in range(tex_height):
                    for y in range(tex_width):
                        r = int(x / tex_height * 192)
                        g = int(y / tex_width * 192)
                        tex_rgb[x, y, :] = [r, g, 128]
                self.initialize_renderer()
            else:
                ex_ob = self.get_ob()
                self.observation_space = Box(low=-np.inf, high=np.inf, shape=ex_ob.shape, dtype=ex_ob.dtype)

        def update_tree(self, tree):
            """Update the XML tree to include maze walls and flow arrows."""
            worldbody = tree.find('.//worldbody')

            # Add walls.
            for i in range(self.maze_map.shape[0]):
                for j in range(self.maze_map.shape[1]):
                    if self.maze_map[i, j] == 1:
                        ET.SubElement(
                            worldbody,
                            'geom',
                            name=f'block_{i}_{j}',
                            pos=f'{j * self._maze_unit - self._offset_x} {i * self._maze_unit - self._offset_y} {self._maze_height / 2 * self._maze_unit}',
                            size=f'{self._maze_unit / 2} {self._maze_unit / 2} {self._maze_height / 2 * self._maze_unit}',
                            type='box',
                            contype='1',
                            conaffinity='1',
                            material='wall',
                        )

            # Adjust floor size.
            center_x = 2 * (self.maze_map.shape[1] - 3)
            center_y = 2 * (self.maze_map.shape[0] - 3)
            size_x = 2 * self.maze_map.shape[1]
            size_y = 2 * self.maze_map.shape[0]
            floor = tree.find('.//geom[@name="floor"]')
            floor.set('pos', f'{center_x} {center_y} 0')
            floor.set('size', f'{size_x} {size_y} 0.2')

            # Add flow arrow geoms on a dense sub-grid within each free cell.
            # Place a 3x3 grid of arrows per cell for better coverage.
            arrow_idx = 0
            sub_n = 3
            offsets = np.linspace(-self._maze_unit / 3, self._maze_unit / 3, sub_n)
            for i in range(self.maze_map.shape[0]):
                for j in range(self.maze_map.shape[1]):
                    if self.maze_map[i, j] == 0:
                        cx, cy = self.ij_to_xy((i, j))
                        for dx in offsets:
                            for dy in offsets:
                                x = cx + dx
                                y = cy + dy
                                flow_vx, flow_vy = self._flow_for_arrows.get_flow(x, y)
                                mag = math.sqrt(flow_vx ** 2 + flow_vy ** 2)
                                if mag < 1e-6:
                                    continue
                                angle = math.atan2(flow_vy, flow_vx)

                                arrow_len = min(mag * 0.55, 1.0)
                                arrow_width = 0.08

                                t = min(mag / 1.8, 1.0)
                                r_col = t
                                b_col = 1.0 - t
                                g_col = 0.2

                                angle_deg = math.degrees(angle)
                                ET.SubElement(
                                    worldbody,
                                    'geom',
                                    name=f'flow_arrow_{arrow_idx}',
                                    type='box',
                                    pos=f'{x} {y} 0.05',
                                    size=f'{arrow_len / 2} {arrow_width / 2} 0.02',
                                    euler=f'0 0 {angle_deg}',
                                    rgba=f'{r_col:.2f} {g_col:.2f} {b_col:.2f} 0.7',
                                    contype='0',
                                    conaffinity='0',
                                )

                                tip_x = x + math.cos(angle) * arrow_len * 0.5
                                tip_y = y + math.sin(angle) * arrow_len * 0.5
                                ET.SubElement(
                                    worldbody,
                                    'geom',
                                    name=f'flow_head_{arrow_idx}',
                                    type='box',
                                    pos=f'{tip_x} {tip_y} 0.05',
                                    size=f'{arrow_width} {arrow_width * 1.2} 0.02',
                                    euler=f'0 0 {angle_deg}',
                                    rgba=f'{r_col:.2f} {g_col:.2f} {b_col:.2f} 0.9',
                                    contype='0',
                                    conaffinity='0',
                                )
                                arrow_idx += 1

            # Add target geom for states-based observation.
            if self._ob_type == 'states':
                ET.SubElement(
                    worldbody,
                    'geom',
                    name='target',
                    type='cylinder',
                    size='.5 .05',
                    pos='0 0 .05',
                    material='target',
                    contype='0',
                    conaffinity='0',
                )

        def set_tasks(self):
            if self._maze_type == 'medium':
                tasks = [
                    [(1, 1), (6, 6)],
                    [self._fixed_init_ij, self._fixed_goal_ij],
                    [(5, 3), (4, 2)],
                    [(6, 5), (6, 1)],
                    [(2, 6), (1, 1)],
                ]
            else:
                raise ValueError(f'Unknown maze type: {self._maze_type}')

            self.task_infos = []
            for i, task in enumerate(tasks):
                self.task_infos.append(
                    dict(
                        task_name=f'task{i + 1}',
                        init_ij=task[0],
                        init_xy=self.ij_to_xy(task[0]),
                        goal_ij=task[1],
                        goal_xy=self.ij_to_xy(task[1]),
                    )
                )

            if self._reward_task_id == 0:
                self._reward_task_id = 1

        def initialize_renderer(self):
            self.custom_renderer = mujoco.Renderer(
                self.model,
                width=self.width,
                height=self.height,
            )
            self.render()

        def reset(self, options=None, *args, **kwargs):
            if options is None:
                options = {}
            if self._reward_task_id is not None:
                assert 1 <= self._reward_task_id <= self.num_tasks
                self.cur_task_id = self._reward_task_id
                self.cur_task_info = self.task_infos[self.cur_task_id - 1]
            elif 'task_id' in options:
                assert 1 <= options['task_id'] <= self.num_tasks
                self.cur_task_id = options['task_id']
                self.cur_task_info = self.task_infos[self.cur_task_id - 1]
            elif 'task_info' in options:
                self.cur_task_id = None
                self.cur_task_info = options['task_info']
            elif self._fixed_start_goal:
                # Fixed mode: always use task 2 (the configured fixed pair).
                self.cur_task_id = 2
                self.cur_task_info = self.task_infos[self.cur_task_id - 1]
            else:
                # Variable mode: sample random free cells for start and goal.
                self.cur_task_id = None
                init_ij = self._free_cells[np.random.randint(len(self._free_cells))]
                goal_ij = self._free_cells[np.random.randint(len(self._free_cells))]
                self.cur_task_info = dict(init_ij=init_ij, goal_ij=goal_ij)

            render_goal = options.get('render_goal', False)

            init_xy = self.add_noise(self.ij_to_xy(self.cur_task_info['init_ij']))
            goal_xy = self.ij_to_xy(self.cur_task_info['goal_ij'])
            if self._add_noise_to_goal:
                goal_xy = self.add_noise(goal_xy)

            super().reset(*args, **kwargs)

            for _ in range(5):
                super().step(self.action_space.sample())

            self.set_goal(goal_xy=goal_xy)
            self.set_xy(goal_xy)
            goal_ob = self.get_oracle_rep() if self._use_oracle_rep else self.get_ob()
            if render_goal:
                goal_rendered = self.render()

            ob, info = super().reset(*args, **kwargs)
            self.set_goal(goal_xy=goal_xy)
            self.set_xy(init_xy)
            ob = self.get_ob()
            info['goal'] = goal_ob
            if render_goal:
                info['goal_rendered'] = goal_rendered

            return ob, info

        def step(self, action):
            if self._success_timing == 'pre':
                success = self.compute_success()

            ob, reward, terminated, truncated, info = super().step(action)

            if self._success_timing == 'post':
                success = self.compute_success()

            action_magnitude = np.linalg.norm(action)
            is_drifting = action_magnitude < self._drift_threshold

            # Base reward: goal achievement.
            if success:
                if self._terminate_at_goal:
                    terminated = True
                info['success'] = 1.0
                reward = self._goal_reward
            else:
                info['success'] = 0.0
                reward = 0.0

            # Energy cost: penalize action magnitude (powered movement).
            energy_cost = -self._energy_weight * action_magnitude
            reward += energy_cost

            # Time cost: constant per-step penalty (every step costs time).
            reward -= self._time_weight

            # Distance cost: per-step penalty proportional to distance from goal.
            dist_to_goal = np.linalg.norm(self.get_xy() - self.cur_goal_xy)
            reward -= self._distance_weight * dist_to_goal

            # Shift for singletask (reward_task_id) environments.
            if self._reward_task_id is not None:
                reward = reward - 1.0

            # Add components to info for debugging/logging.
            info['energy_cost'] = energy_cost
            info['is_drifting'] = float(is_drifting)
            info['dist_to_goal'] = dist_to_goal

            return ob, reward, terminated, truncated, info

        def render(self):
            if self.custom_renderer is None:
                self.initialize_renderer()
            self.custom_renderer.update_scene(self.data, camera=self.custom_camera)
            return self.custom_renderer.render()

        def get_ob(self, ob_type=None):
            ob_type = self._ob_type if ob_type is None else ob_type
            if ob_type == 'states':
                base_ob = super().get_ob()  # [qpos_x, qpos_y, flow_vx, flow_vy]
                return np.concatenate([base_ob, self.cur_goal_xy])
            else:
                return self.render()

        def get_oracle_rep(self):
            return np.array(self.cur_goal_xy)

        def compute_success(self):
            return np.linalg.norm(self.get_xy() - self.cur_goal_xy) <= self._goal_tol

        def set_goal(self, goal_ij=None, goal_xy=None):
            if goal_xy is None:
                self.cur_goal_xy = self.ij_to_xy(goal_ij)
                if self._add_noise_to_goal:
                    self.cur_goal_xy = self.add_noise(self.cur_goal_xy)
            else:
                self.cur_goal_xy = goal_xy
            if self._ob_type == 'states':
                self.model.geom('target').pos[:2] = self.cur_goal_xy

        def get_oracle_subgoal(self, start_xy, goal_xy):
            start_ij = self.xy_to_ij(start_xy)
            goal_ij = self.xy_to_ij(goal_xy)

            bfs_map = self.maze_map.copy()
            for i in range(self.maze_map.shape[0]):
                for j in range(self.maze_map.shape[1]):
                    bfs_map[i][j] = -1

            bfs_map[goal_ij[0], goal_ij[1]] = 0
            queue = [goal_ij]
            while len(queue) > 0:
                i, j = queue.pop(0)
                for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if (
                        0 <= ni < self.maze_map.shape[0]
                        and 0 <= nj < self.maze_map.shape[1]
                        and self.maze_map[ni, nj] == 0
                        and bfs_map[ni, nj] == -1
                    ):
                        bfs_map[ni][nj] = bfs_map[i][j] + 1
                        queue.append((ni, nj))

            subgoal_ij = start_ij
            for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                ni, nj = start_ij[0] + di, start_ij[1] + dj
                if (
                    0 <= ni < self.maze_map.shape[0]
                    and 0 <= nj < self.maze_map.shape[1]
                    and self.maze_map[ni, nj] == 0
                    and bfs_map[ni, nj] < bfs_map[subgoal_ij[0], subgoal_ij[1]]
                ):
                    subgoal_ij = (ni, nj)
            subgoal_xy = self.ij_to_xy(subgoal_ij)
            return np.array(subgoal_xy), bfs_map

        def xy_to_ij(self, xy):
            maze_unit = self._maze_unit
            i = int((xy[1] + self._offset_y + 0.5 * maze_unit) / maze_unit)
            j = int((xy[0] + self._offset_x + 0.5 * maze_unit) / maze_unit)
            return i, j

        def ij_to_xy(self, ij):
            i, j = ij
            x = j * self._maze_unit - self._offset_x
            y = i * self._maze_unit - self._offset_y
            return x, y

        def add_noise(self, xy):
            random_x = np.random.uniform(low=-self._noise, high=self._noise) * self._maze_unit / 4
            random_y = np.random.uniform(low=-self._noise, high=self._noise) * self._maze_unit / 4
            return xy[0] + random_x, xy[1] + random_y

    return ZermeloMazeEnv(*args, **kwargs)
