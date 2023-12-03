import gymnasium as gym
from gymnasium.spaces import Discrete, Dict, Box
import numpy as np

import pdb

class envWrapper(gym.Env):
    def __init__(self, RDDLEnv, sample_range, max_episode_length=100, random_start=False):
        super(envWrapper, self).__init__()
        self.RDDLEnv = RDDLEnv
        
        self.action_list = sorted(self.RDDLEnv.action_space.keys())
        self.observation_list = sorted(self.RDDLEnv.observation_space.keys())

        self.action_space = Discrete(len(self.action_list))
        self.observation_space = self.new_observation_space(self.RDDLEnv)

        self.state = RDDLEnv.reset()

        self.max_episode_length = max_episode_length
        self.random_start = random_start

        self.current_step = 0

        self.sample_range = sample_range
    
    def convert_dict_numpy(self, int_dict):
        np_dict = {}
        for k,v in int_dict.items():
            if isinstance(v, np.bool_):
                np_dict[k] = int(v)
            else:
                np_dict[k] = np.array([v], dtype=np.float32)
        return np_dict


    def action_vec2dict(self, action_num):
        action_dict = {}
        for idx, action_str in enumerate(self.action_list):
            v = 1 if idx == action_num else 0
            action_dict[action_str] = v
        return action_dict

    def action_dict2vec(self, action_dict):
        action_vec = np.zeros(len(self.action_list))
        for idx, action in enumerate(self.action_list):
            action_vec[idx] = action_dict[action]
        return action_vec
    
    def new_observation_space(self, RDDLEnv):
        new_observation_space = Dict()

        for k,v in RDDLEnv.observation_space.items():            
            if 'gym.spaces.box.Box' in str(type(v)):
                low = v.low
                high = v.high
                shape = v.shape
                new_observation_space[k] = Box(low=low,high=high,shape=shape,dtype=np.float32)
            elif 'gym.spaces.discrete.Discrete' in str(type(v)):
                new_observation_space[k] = Discrete(v.n)
        return new_observation_space

    def init_random_rddl_state(self, state, seed):
        self.RDDLEnv.state = state
        self.RDDLEnv.sampler.state = state
        self.RDDLEnv.sampler.obs = state
        for k, v in state.items():
            name = k.split('___')[0]
            idx = int(k.split('___')[1][1:])-1
            self.RDDLEnv.sampler.init_values[name][idx] = v
        self.RDDLEnv.reset(seed)

    def reset(self, seed=None):
        if self.random_start:
            state = {}
            for k, v in self.RDDLEnv.observation_space.items():
                if 'gym.spaces.box.Box' in str(type(v)):
                    low = self.sample_range[k][0]
                    high = self.sample_range[k][1]
                    interval = self.sample_range[k][2]
                    if self.sample_range[k][2] == None:
                        sample = np.random.uniform(low=low, high=high)
                    else:
                        arange = np.arange(low, high+interval, interval)
                        sample = np.random.choice(arange) 
                    state[k] = np.float64(sample)
                else:
                    sample = np.random.choice([True, False])
                    state[k] = sample
            self.state = self.convert_dict_numpy(state)
            self.init_random_rddl_state(state, seed)
        else:
            self.state = self.convert_dict_numpy(self.RDDLEnv.reset(seed))

        self.current_step = 0
        return (self.state, {})
    
    def step(self, action):

        action_dict = self.action_vec2dict(action)
        next_state, reward, done, info = self.RDDLEnv.step(action_dict)

        self.current_step += 1

        if self.current_step >= self.max_episode_length:
            done = True
        else:
            done = False

        self.state = self.convert_dict_numpy(next_state)  

        return self.state, reward, done, False, info  # Modify as needed

