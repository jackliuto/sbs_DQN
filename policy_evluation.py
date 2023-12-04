from pyRDDLGym import RDDLEnv

from value_generator.generators import ValueGenerator
from envs.Env  import envWrapper
from stable_baselines3 import DQN

import pdb


DOMAIN_PATH_TARGET = "./RDDL/mars_rover/domain.rddl"
INSTANCE_PATH_TARGET = "./RDDL/mars_rover/instance2.rddl"
DOMAIN_PATH_SOURCE = "./RDDL/mars_rover/domain.rddl"
INSTANCE_PATH_SOURCE = "./RDDL/mars_rover/instance1.rddl"

MAX_DEPTH = None
N_PE_STEPS = 10


MODEL_PATH = '../checkpoints/mars_rover/rl_model_8368000_steps.zip'
SAVE_PATH = './saved_tensor/rover_{}.npy'.format(N_PE_STEPS)
SAMPLE_RANGE = {'pos_x___a1':(0.0,10.0,1), 'pos_y___a1':(0.0,10.0,1), 'has_mineral___a1':(True, False, None)}


RDDLEnv = RDDLEnv.RDDLEnv(domain=DOMAIN_PATH_TARGET, instance=INSTANCE_PATH_TARGET)
env = envWrapper(RDDLEnv, sample_range=SAMPLE_RANGE, max_episode_length=100)
loaded_model = DQN.load(MODEL_PATH, 
                        env=env, exploration_fraction=0.0)

value_generator = ValueGenerator(env, 
                                 domain_path_source=DOMAIN_PATH_SOURCE, instance_path_source=INSTANCE_PATH_SOURCE,
                                 domain_path_target=DOMAIN_PATH_TARGET, instance_path_target=INSTANCE_PATH_TARGET,
                                 network=loaded_model, 
                                 save_path = SAVE_PATH,
                                 sample_range=SAMPLE_RANGE, 
                                 max_depth=MAX_DEPTH, 
                                 n_pe_steps=N_PE_STEPS)

# value_generator.do_pe()