from pyRDDLGym import RDDLEnv

from value_generator.generators import ValueGenerator
from envs.Env  import envWrapper
from stable_baselines3 import DQN

import pdb


DOMAIN_PATH = "./RDDL/mars_rover/domain.rddl"
INSTANCE_PATH = "./RDDL/mars_rover/instance1.rddl"

MAX_DEPTH = 10

SAMPLE_RANGE = {'pos_x___a1':(0.0,10.0), 'pos_y___a1':(0.0,10.0), 'has_mineral___a1':(True, False)}


RDDLEnv = RDDLEnv.RDDLEnv(domain=DOMAIN_PATH, instance=INSTANCE_PATH)
env = envWrapper(RDDLEnv, sample_range=SAMPLE_RANGE, max_episode_length=100)
loaded_model = DQN.load('../checkpoints/mars_rover/rl_model_8368000_steps.zip', env=env, exploration_fraction=0.0)

value_generator = ValueGenerator(env, domain_path=DOMAIN_PATH, instance_path=INSTANCE_PATH,
                                 network=loaded_model, sample_range=SAMPLE_RANGE, max_depth=MAX_DEPTH)

value_generator.network2tree()