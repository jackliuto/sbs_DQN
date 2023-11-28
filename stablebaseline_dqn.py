import warnings
warnings.filterwarnings("ignore")

from pyRDDLGym import RDDLEnv

from envs.Env  import envWrapper
from dqn.LowerboundDQN import LoweboundDQN



from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure

from stable_baselines3.common.env_checker import check_env

# globalvars
DOMAIN_PATH = "./RDDL/mars_rover/domain.rddl"
INSTANCE_PATH = "./RDDL/mars_rover/instance1.rddl"

GAMMA = 0.9
TAU = 0.001
EXP_BEG = 0.9
EXP_FINAL = 0.05
BUFFER_SIZE = 1000000
BATCH_SIZE = 64
EXPLORATION_FRACTION = 0.3

MAX_EPS_LEN = 1000

RANDOM_START = True

total_timesteps = 100000000    # Adjust as needed

SAMPLE_RANGE = {'pos_x___a1':(0.0,10.0), 'pos_y___a1':(0.0,10.0), 'has_mineral___a1':(True, False)}


# set enviroment
RDDLEnv = RDDLEnv.RDDLEnv(domain=DOMAIN_PATH, instance=INSTANCE_PATH)
env = envWrapper(RDDLEnv, sample_range=SAMPLE_RANGE, max_episode_length=MAX_EPS_LEN, random_start=RANDOM_START)

check_env(env)


# set callbacks
checkpoint_callback = CheckpointCallback(save_freq = 1000, 
                                         save_path = './checkpoints/rover_random_start/',
                                         name_prefix = 'rl_model')

# eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
#                              log_path='./logs/', eval_freq=500,
#                              deterministic=True, render=False)

# set up logger
tmp_path = "./logs/rover_random_start/"
new_logger = configure(tmp_path, ["stdout", "csv", "log", "tensorboard"])                    

model = LoweboundDQN('MultiInputPolicy', env, 
            gamma=GAMMA,
            tau=TAU,
            exploration_initial_eps=EXP_BEG,
            exploration_final_eps=EXP_FINAL,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            exploration_fraction=EXPLORATION_FRACTION,
            verbose=1)

model.set_logger(new_logger)

for i in range(10):
    print(env.reset())
    action = env.action_space.sample()
    state = env.step(env.action_space.sample())
    print(action, state)
    print(env.RDDLEnv.state)

raise ValueError


training_info = model.learn(total_timesteps=total_timesteps,
                            callback=checkpoint_callback)

