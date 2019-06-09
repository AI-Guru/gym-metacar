import gym
try:
    import gym_metacar
except:
    import sys
    sys.path.append(".")
    sys.path.append("..")
    import gym_metacar

env = gym.make("metacar-level2-discrete-v0")
env.reset()
print(env.observation_space)
print(env.action_space)

for step in range(100):
    print(step)
    observation, _, _, _ = env.step(env.action_space.sample())
    env.render()
env.close()
