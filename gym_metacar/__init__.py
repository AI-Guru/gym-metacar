from gym.envs.registration import register
import itertools

def register_metacar_environment(level, action_space):
    id = "metacar-{}-{}-v0".format(level, action_space)
    register(
        id=id,
        entry_point='gym_metacar.envs:MetacarEnv',
        kwargs={"level" : level, "discrete" : action_space == "discrete"}
    )
    print("Registered environment \"{}\"".format(id))

# Register all environments.
levels = ["level0", "level1", "level2", "level3" , "random"]
action_spaces = ["discrete", "continuous"]
for level, action_space in itertools.product(levels, action_spaces):
    register_metacar_environment(level, action_space)
