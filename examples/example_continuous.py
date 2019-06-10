import gym
try:
    import gym_metacar
except:
    import sys
    sys.path.append(".")
    sys.path.append("..")
    import gym_metacar

env = gym.make("metacar-random-continuous-v0")
env.enable_webrenderer()
env.reset()
print(env.observation_space)
print(env.action_space)

for step in range(1000):
    env.render()
    observation, _, _, _ = env.step(env.action_space.sample())
    print(step, observation)

env.close()
