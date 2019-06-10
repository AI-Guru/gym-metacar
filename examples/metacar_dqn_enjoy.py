import gym
try:
    import gym_metacar
except:
    import sys
    sys.path.append(".")
    sys.path.append("..")
    import gym_metacar

from gym_metacar.wrappers import *

from stable_baselines import DQN
from stable_baselines.common.vec_env import *

env_id = "metacar-random-discrete-v0"
env = gym.make(env_id)
env.enable_webrenderer()
env = LinearObservationWrapper(env)
env = TerminateWrapper(env)
env = ClipRewardsWrapper(env)
env = DummyVecEnv([lambda:env])
env = VecFrameStack(env, n_stack=4)

# Load the trained agent
model = DQN.load("metacar-dqn")

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done == True:
        env.reset()
        continue
    env.render()

env.close()
