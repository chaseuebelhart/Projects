from gym.envs.registration import registry, register, make, spec
from gym_tetris_raw.tetris_env import TetrisEnv
# Pygame
# ----------------------------------------
for game in ['Tetris']:
    nondeterministic = False
    register(
        id='{}-v1'.format(game),
        entry_point='gym_tetris_raw:TetrisEnv',
        kwargs={},
        timestep_limit=10000,
        nondeterministic=nondeterministic,
    )
