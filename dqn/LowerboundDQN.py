
import torch as th
import numpy as np

from stable_baselines3.dqn import DQN

from torch.nn import functional as F

from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import pdb

class LowerboundDQN(DQN):
    def __init__(self, policy, env, lowerbound_path, warmstart_path, algo_type, **kwargs):
        super(LowerboundDQN, self).__init__(policy, env, **kwargs)
        self.algo_type = algo_type
        self.action_list = env.action_list
        self.observation_list = env.observation_list
        self.sample_range = env.sample_range
        if 'lowerbound' in self.algo_type:
            self.lower_bound_tensor = th.tensor(np.load(lowerbound_path)).to(self.device)
        self.warm_start_path = warmstart_path

        self.lb_decay = 0.999

        self.ppr_rate = 0.1

        if 'warmstart' in self.algo_type:
            self.source_model = DQN.load(self.warm_start_path)
            q_net_state_dict = self.source_model.policy.q_net.state_dict()
            q_net_target_state_dict = self.policy.q_net_target.state_dict()
            self.policy.q_net.load_state_dict(q_net_state_dict)
            self.policy.q_net_target.load_state_dict(q_net_target_state_dict)

        if "ppr" in self.algo_type:
            self.ppr_model = DQN.load(self.warm_start_path)
            

    def state_to_lowerbound(self, state, lb_tensor):
        combined_tensor = th.empty((self.batch_size, 1), dtype=th.float32).to(self.device)
        for s in self.observation_list:
            interval = self.sample_range[s][2]
            if interval != None:
                state_tensor = state[s]
                rounded_state_tensor = th.round(state_tensor / interval) * interval
                idx_state_tensor = rounded_state_tensor / interval
                combined_tensor = th.cat((combined_tensor, idx_state_tensor), dim=1)
            else:
                combined_tensor = th.cat((combined_tensor, state[s]), dim=1)
        idx_tensor = combined_tensor[:,1:].to(th.int64)
        indices = list(idx_tensor.t())
        lowerbound = lb_tensor[indices].reshape(-1, 1)

        return lowerbound



    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

                if 'lowerbound' in self.algo_type:
                    lowerbound_v_values = self.state_to_lowerbound(replay_data.next_observations, self.lower_bound_tensor)
                    lowerbound_v_values *= self.lb_decay
                    lowerbound_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * lowerbound_v_values
                    max_q_values = th.max(target_q_values, lowerbound_q_values)
                    target_q_values = max_q_values
                
            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[next(iter(observation.keys()))].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            if self.algo_type == "ppr":
                if np.random.rand() < self.ppr_rate:
                    action, state = self.ppr_model.policy.predict(observation, state, episode_start, deterministic)
                else:
                    action, state = self.policy.predict(observation, state, episode_start, deterministic)
            else :
                action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state