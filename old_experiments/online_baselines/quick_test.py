# quick_test2.py
import os
os.environ["MUJOCO_GL"] = "osmesa"

import gym
import d4rl
import mujoco_py

# Check if mujoco_py can render
env = gym.make("antmaze-umaze-v2")
env.reset()

# Try the sim directly
sim = env.unwrapped.sim
viewer = mujoco_py.MjRenderContextOffscreen(sim, device_id=-1)
viewer.render(640, 480)
frame = viewer.read_pixels(640, 480, depth=False)

print(f"Success! Shape: {frame.shape}")