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
from stable_baselines.ddpg.policies import *
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.vec_env import *

env_id = "metacar-random-continuous-v0"
env = gym.make(env_id)
env = LinearObservationWrapper(env)
env = TerminateWrapper(env)
env = ClipRewardsWrapper(env)
env = DummyVecEnv([lambda:env])
env = VecFrameStack(env, n_stack=4)

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

# Create the agent.
tensorboard_log = "logs/metacar-ddpg"
model = DDPG(
    MlpPolicy,
    env=env,
    param_noise=param_noise,
    action_noise=action_noise,
    verbose=1,
    tensorboard_log=tensorboard_log)

model.learn(total_timesteps=1000000)
model.save("metacar-ddpg")
