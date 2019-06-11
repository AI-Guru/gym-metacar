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
from stable_baselines.deepq.policies import *
from stable_baselines.common.vec_env import *

env_id = "metacar-random-discrete-v0"
env = gym.make(env_id)
env = LinearObservationWrapper(env)
env = TerminateWrapper(env)
env = ClipRewardsWrapper(env)
env = DummyVecEnv([lambda:env])
env = VecFrameStack(env, n_stack=4)

# Create the agent.
tensorboard_log = "logs/metacar-dqn"
model = DQN(
    MlpPolicy,
    env=env,
    #learning_rate=0.00025,
    target_network_update_freq=10000,
    learning_starts=100000,
    buffer_size=100000,
    verbose=1, tensorboard_log=tensorboard_log)

model.learn(total_timesteps=2000000)
model.save("metacar-dqn")
