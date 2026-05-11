"""Zermelo navigation environment — point agent in a maze with fluid flow."""

from gymnasium.envs.registration import register

register(
    id='zermelo-pointmaze-medium-v0',
    entry_point='zermelo_env.zermelo_maze:make_zermelo_maze_env',
    max_episode_steps=1000,
    kwargs=dict(maze_type='medium'),
)

register(
    id='zermelo-pointarena-medium-v0',
    entry_point='zermelo_env.zermelo_maze:make_zermelo_maze_env',
    max_episode_steps=1000,
    kwargs=dict(maze_type='medium', maze_on=False),
)

singletask_dict = dict(
    add_noise_to_goal=False,
    success_timing='pre',
)

for task_id in [None, 1, 2, 3, 4, 5]:
    task_suffix = '' if task_id is None else f'-task{task_id}'
    reward_task_id = 0 if task_id is None else task_id
    register(
        id=f'zermelo-pointmaze-medium-singletask{task_suffix}-v0',
        entry_point='zermelo_env.zermelo_maze:make_zermelo_maze_env',
        max_episode_steps=1000,
        kwargs=dict(
            maze_type='medium',
            reward_task_id=reward_task_id,
            **singletask_dict,
        ),
    )
