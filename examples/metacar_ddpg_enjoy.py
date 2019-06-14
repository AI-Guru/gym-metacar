import gym
try:
    import gym_metacar
except:
    import sys
    sys.path.append(".")
    sys.path.append("..")
    import gym_metacar

from gym_metacar.wrappers import *

from stable_baselines import DDPG
from stable_baselines.common.vec_env import *

env_id = "metacar-random-continuous-v0"
env = gym.make(env_id)
env.enable_webrenderer()
env = LinearObservationWrapper(env)
env = TerminateWrapper(env)
env = StepLimitTerminateWrapper(env, 1000)
env = ClipRewardsWrapper(env)
env = DummyVecEnv([lambda:env])
env = VecFrameStack(env, n_stack=4)

# Load the trained agent
model = DDPG.load("metacar-ddpg")

# Enjoy trained agent
for episode in range(10):
    print("Episode ", episode)
    obs = env.reset()
    done = False
    while done == False:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()

env.close()
