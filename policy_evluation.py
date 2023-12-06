from pyRDDLGym import RDDLEnv

from value_generator.generators import ValueGenerator
from envs.Env  import envWrapper
from stable_baselines3 import DQN

from utils.params import Params

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


params = Params("./params/sdp/rover.json")

# globalvars

DOMAIN_TYPE = params.domain_type
DOMAIN_PATH_SOURCE = params.domain_path_source
DOMAIN_PATH_TARGET = params.domain_path_target
INSTANCE_PATH_SOURCE = params.instance_path_source
INSTANCE_PATH_TARGET = params.instance_path_target
if params.tree_depth == 0:
    TREE_DEPTH = None
else:
    TREE_DEPTH = params.tree_depth
SDP_STEPS = params.sdp_steps
MODEL_PATH=params.model_path

MODEL_NAME = '{0}_{1}_{2}'.format(DOMAIN_TYPE, SDP_STEPS, TREE_DEPTH)
SAVE_PATH = params.save_path+MODEL_NAME+'.npy'

if params.domain_type == "rover":
    SAMPLE_RANGE = {'pos_x___a1':(0.0, 10.0, 1.0), 'pos_y___a1':(0.0, 10.0, 1.0), 'has_mineral___a1':(True, False, None)}
elif params.domain_type == "uav":
    SAMPLE_RANGE = {'pos_x___a1':(0.0, 10.0, 1.0), 'pos_y___a1':(0.0, 10.0, 1.0), 'pos_z___a1':(0.0, 10.0, 1.0),
                    'vel___a1':(1.0, 2.0, 0.1),
                    'phi___a1':(0.0, 1.0, 0.1), 'theta___a1':(0.0, 1.0, 0.1), 'psi___a1':(0.0, 1.0, 0.1),
                    }


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
                                 max_depth=TREE_DEPTH, 
                                 n_pe_steps=SDP_STEPS)
