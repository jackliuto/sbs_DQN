import warnings
warnings.filterwarnings("ignore")

import os

import wandb
from wandb.integration.sb3 import WandbCallback

from envs.Env  import envWrapper
from dqn.LowerboundDQN import LowerboundDQN
from utils.params import Params

from pyRDDLGym import RDDLEnv
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_checker import check_env


params = Params("./params/train/power.json")

# globalvars
ALGO_TYPE = params.algo_type
DOMAIN_PATH = params.domain_path
if params.instance_type == "source":
    INSTANCE_PATH = params.instance_path_source
elif params.instance_type == "target":
    INSTANCE_PATH = params.instance_path_target
WS_PATH = params.warmstart_model
GAMMA = params.gamma
TAU = params.tau
EXP_BEG = params.exp_beg
EXP_FINAL = params.exp_final
BUFFER_SIZE = params.buffer_size
BATCH_SIZE = params.batch_size
EXPLORATION_FRACTION = params.exploration_fraction
MAX_EPS_LEN = params.max_eps_length
TOTAL_TIMESTEPS = params.total_timesteps
RANDOM_START = params.random_start
LB_PATH = params.cache_path
TAREGET_UPDATE_INTERVAL = params.target_update_interval
MODEL_NAME = '{0}_{1}_{2}_{3}_{4}_{5}_{6}'.format(params.domain_type, params.sdp_steps, params.algo_type, 
                                              params.instance_type, params.target_update_interval,
                                              params.exp_beg,
                                              params.notes)

SAVE_PATH = params.save_path+MODEL_NAME+'/'
LOG_PATH = params.log_path+MODEL_NAME+'/'


if "rover" in params.domain_type :
    SAMPLE_RANGE = {'pos_x___a1':(0.0, 10.0, 1.0), 'pos_y___a1':(0.0, 10.0, 1.0), 'has_mineral___a1':(True, False, None)}
elif params.domain_type == "uav":
    SAMPLE_RANGE = {'pos_x___a1':(0.0, 10.0, 1.0), 'pos_y___a1':(0.0, 10.0, 1.0), 'pos_z___a1':(0.0, 10.0, 1.0),
                    'vel___a1':(1.0, 2.0, 0.1),
                    'phi___a1':(0.0, 1.0, 0.1), 'theta___a1':(0.0, 1.0, 0.1), 'psi___a1':(0.0, 1.0, 0.1),
                    }
elif params.domain_type == "reservoir":
    SAMPLE_RANGE = {'rlevel___t1':(0.0, 100.0, 1.0), 'rlevel___t2':(0.0, 100.0, 1.0)}
elif params.domain_type == "power":
    SAMPLE_RANGE = {'curProd___p1':(0.0, 10, 1.0), 'curProd___p2':(0.0, 10.0, 1.0),'curProd___p3':(0.0, 10.0, 1.0)}


policy_kwargs = dict(
    net_arch=params.net_arch  # Example architecture: three layers with 64, 128, and 64 units
)

wandb.init(project="lb_dqn_CC", 
           name=MODEL_NAME, 
           config=params,
           sync_tensorboard=True
           )


# set enviroment
RDDLEnv = RDDLEnv.RDDLEnv(domain=DOMAIN_PATH, instance=INSTANCE_PATH)
env = envWrapper(RDDLEnv, sample_range=SAMPLE_RANGE, max_episode_length=MAX_EPS_LEN, random_start=RANDOM_START)


eval_env = envWrapper(RDDLEnv, sample_range=SAMPLE_RANGE, max_episode_length=MAX_EPS_LEN, random_start=False)

# set callbacks
checkpoint_callback = CheckpointCallback(save_freq = 1000000, 
                                         save_path = SAVE_PATH,
                                         name_prefix = MODEL_NAME)

eval_callback = EvalCallback(eval_env,
                             best_model_save_path=SAVE_PATH,
                             log_path=LOG_PATH,
                             eval_freq=1000,  # Evaluate every 1000 iterations
                             deterministic=True,
                             render=False)

# set up logger
new_logger = configure(LOG_PATH, ["stdout", "csv", "log", "tensorboard"])        

model = LowerboundDQN(
            'MultiInputPolicy', 
            env,
            LB_PATH,
            WS_PATH,
            ALGO_TYPE,
            gamma=GAMMA,
            tau=TAU,
            exploration_initial_eps=EXP_BEG,
            exploration_final_eps=EXP_FINAL,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            exploration_fraction=EXPLORATION_FRACTION,
            target_update_interval=TAREGET_UPDATE_INTERVAL,
            verbose=0,
            policy_kwargs=policy_kwargs)

model.set_logger(new_logger)

# for i in range(100):
#     s = env.reset()
#     print(s)
#     for i in range(100):
#         action = env.action_space.sample()
#         ns, r, _, _, _ = env.step(action)
#         # if r == 1:
#         #     print(s)
#         s = ns

# raise ValueError

WandbCallback = WandbCallback()

training_info = model.learn(total_timesteps=TOTAL_TIMESTEPS,
                            callback=CallbackList([checkpoint_callback, eval_callback, WandbCallback]))



