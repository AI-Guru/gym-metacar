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
        observation = np.array(observation["linear"], dtype="float32")
        return observation


class LidarObservationWrapper(gym.ObservationWrapper):
    """
    Yields lidar observation data only.
    """

    def __init__(self, env, for_cnn=False):
        super(LidarObservationWrapper, self).__init__(env)

        self.for_cnn = for_cnn

        self.observation_space = env.observation_space["lidar"]
        if for_cnn == True:
            self.observation_space = gym.spaces.Box(
                low=self.observation_space.low[0][0],
                high=self.observation_space.high[0][0],
                shape=self.observation_space.shape + (1,),
                dtype=self.observation_space.dtype,
            )

    def observation(self, observation):
        observation = np.array(observation["lidar"], dtype="float32")
        if self.for_cnn == True:
            observation = np.expand_dims(observation, axis=-1)
        return observation


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
        return observation, reward, done, info

class StepLimitTerminateWrapper(gym.Wrapper):
    """
    Stops the simulation when steps limit exceeded.
    """

    def __init__(self, env, step_limit):
        super(StepLimitTerminateWrapper, self).__init__(env)

        self.step_limit = step_limit
        self.current_step = 0

    def step(self, action):
        self.current_step += 1

        observation, reward, done, info = super().step(action)
        if self.current_step >= self.step_limit:
            done = True
        return observation, reward, done, info

    def reset(self):
        self.current_step = 0
        return super().reset()


class ClipRewardsWrapper(gym.RewardWrapper):
    """
    Clips the rewards.
    """

    def __init__(self, env):
        super(ClipRewardsWrapper, self).__init__(env)

    def reward(self, reward):
        return np.clip(reward, -1., 1.)
