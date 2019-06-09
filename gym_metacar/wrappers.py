import gym
import numpy as np

class LinearObservationWrapper(gym.ObservationWrapper):
    """
    Yields linear observation data only.
    """

    def __init__(self, env):
        super(LinearObservationWrapper, self).__init__(env)
        self.observation_space = env.observation_space["linear"]

    def observation(self, observation):
        return observation["linear"]


class LidarObservationWrapper(gym.ObservationWrapper):
    """
    Yields lidar observation data only.
    """

    def __init__(self, env):
        super(LidarObservationWrapper, self).__init__(env)
        self.observation_space = env.observation_space["lidar"]

    def observation(self, observation):
        return observation["lidar"]


class TerminateWrapper(gym.Wrapper):
    """
    Stops the simulation when the reward is -1.
    """

    def __init__(self, env):
        super(TerminateWrapper, self).__init__(env)

    def step(self, action):
        observation, reward, done, info = super().step(action)
        if reward == -1:
            done = True
            print("Simulation is done.")
        return observation, reward, done, info


class ClipRewardsWrapper(gym.RewardWrapper):
    """
    Clips the rewards.
    """

    def __init__(self, env):
        super(ClipRewardsWrapper, self).__init__(env)

    def reward(self, reward):
        return np.clip(reward, -1., 1.)
