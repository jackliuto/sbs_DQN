import torch as th
import torch.nn as nn

import warnings

from gymnasium.spaces import Discrete, Dict, Box

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import tree

from matplotlib import pyplot as plt

from pyRDDLGym.Core.Grounder.RDDLGrounder import RDDLGrounder
from pyRDDLGym.Core.Parser.parser import RDDLParser
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD

# from pySDP.core.mdp import MDP
# from pySDP.core.parser import Parser
# from pySDP.core.policy import Policy
# from pySDP.policy_evaluation.pe import PolicyEvaluation

from pyRDDLGym.Solvers.SDP.helper.mdp import MDP
from pyRDDLGym.Solvers.SDP.helper.mdp_parser import Parser
from pyRDDLGym.Solvers.SDP.helper.policy import Policy
from pyRDDLGym.Solvers.SDP.pe import PolicyEvaluation

import pdb

class ValueGenerator:
    def __init__(self, env, 
                 domain_path_source, instance_path_source,
                 domain_path_target, instance_path_target, 
                 network,
                 save_path, 
                 sample_range, 
                 max_depth=None,
                 n_pe_steps=5, 
                 discount=0.9):
        
        self.sample_size = 100000

        self.env = env
        self.network = network
        self.save_path = save_path
        self.sample_range = sample_range
        self.max_depth = max_depth
        self.n_pe_steps = n_pe_steps
        self.disount = discount
        self.xadd_model_target, self.xadd_context_target = self.get_xadd_model_from_file(domain_path_target, instance_path_target)
        self.xadd_model_source, self.xadd_context_source = self.get_xadd_model_from_file(domain_path_source, instance_path_source)
        self.diff_reward_node = self.build_model_diff()
        self.decision_tree_nn = self.network2tree(max_depth=max_depth)
        self.policy_xadd_dict = self.tree2xaddpolicy(self.decision_tree_nn)
        self.xadd_value_node = self.do_pe(self.xadd_model_target, self.xadd_context_target, self.policy_xadd_dict, self.diff_reward_node)
        self.xadd_tensor = self.xadd2tensor(self.xadd_context_target, self.xadd_value_node)

        

    
    # def regress_reward(self, xadd, reward_id):

        
    def build_model_diff(self):
        
        source_reward_node = self.xadd_context_source._id_to_node.get(self.xadd_model_source.reward, None)
        source_reward_node.turn_off_print_node_info()
        node_str = str(source_reward_node)
        source_reward_id = self.xadd_context_target.import_xadd(xadd_str=node_str, locals=self.xadd_model_target.ns)
        diff_reward_id = self.xadd_context_target.apply(self.xadd_model_target.reward, source_reward_id, 'subtract')

        diff_reward_id = self.xadd_context_target.reduce_lp(diff_reward_id)

        return diff_reward_id

    
    def do_pe(self, model, context, policy_xadd_dict, reward_id):

        model.reward = reward_id

        parser = Parser()
        mdp =  parser.parse(model, discount=self.disount, is_linear=True)
        policy = Policy(mdp)

        policy_dict = {}
        for aname, action in mdp.actions.items():
            policy_dict[action] = policy_xadd_dict[aname]
        policy.load_policy(policy_dict)

        pe = PolicyEvaluation(mdp, self.n_pe_steps)
        pe.load_policy(policy)

        value_id_pe = pe.solve()

        return value_id_pe

        # x = np.arange(0, 11, 1)
        # y = np.arange(0, 11, 1)

        # X, Y = np.meshgrid(x, y)
        # Z = np.zeros_like(X, dtype=float)


        # var_set = self.xadd_context.collect_vars(value_id_pe)


        # var_dict = {}
        # for i in var_set:
        #     var_dict[f"{i}"] = i

        # for i in range(len(x)):
        #     for j in range(len(y)):
        #         i_v = i
        #         j_v = j
        #         cont_assign = {var_dict["pos_x___a1"]: i_v, var_dict["pos_y___a1"]: j_v}
        #         bool_assign = {var_dict["has_mineral___a1"]: True}
        #         value = self.xadd_context.evaluate(value_id_pe, bool_assign=bool_assign, cont_assign=cont_assign)

        #         # value = context.evaluate(q_dict['move_east___a1'], bool_assign=bool_assign, cont_assign=cont_assign)
        #         # value = context.evaluate(q_dict['do_nothing___a1'], bool_assign=bool_assign, cont_assign=cont_assign)
        #         Z[i][j] = value

        # np.set_printoptions(precision=2)
        # print(Z.T)

    def xadd2tensor(self, xadd, value_id_pe):
        dim_list = []
        bool_dims = []
        for i, k in enumerate(self.env.observation_list):
            range = self.sample_range[k]
            if isinstance(range[0], bool):
                dim_list.append(2)
                bool_dims.append(i)
            else:
                arange = np.arange(range[0], range[1]+range[2], range[2])
                dim_list.append(len(arange))

        value_tensor = np.zeros(dim_list, dtype=np.float32)
        value_diff_tensor = np.zeros(dim_list, dtype=np.float32)
        value_source_tensor = np.zeros(dim_list, dtype=np.float32)

        indices = list(np.ndindex(tuple(dim_list)))

        var_set = xadd.collect_vars(value_id_pe)
        var_dict = {}
        for i in var_set:
            var_dict[f"{i}"] = i

        for idx in indices:
            bool_assign = {}
            cont_assign = {}
            state = {}
            for i, k in enumerate(self.env.observation_list):
                if i in bool_dims:
                    bool_assign[var_dict[k]] = bool(idx[i])
                    state[k] = th.tensor([[idx[i]]], dtype=th.float32).to(self.network.device)
                else:
                    cont_assign[var_dict[k]] = float(idx[i]*self.sample_range[k][2])
                    state[k] = th.tensor([[idx[i]]], dtype=th.int32).to(self.network.device)

            value_diff = xadd.evaluate(value_id_pe, bool_assign=bool_assign, cont_assign=cont_assign)
            q_values = self.network.policy.q_net(state)
            max_q_values = q_values.max(dim=1)
            value_source = max_q_values.values.item()

            value = value_diff + value_source
            value_tensor[idx] = value

            value_diff_tensor[idx] = value_diff
            value_source_tensor[idx] = value_source

        
        np.set_printoptions(precision=2)


        print('V_target')
        print(value_tensor[0])
        print(value_tensor[1])
        print('V_source')
        print(value_source_tensor[0])
        print(value_source_tensor[1])
        print('Vdiff')
        print(value_diff_tensor[0])
        print(value_diff_tensor[1])
        

        if self.save_path != None:
            np.save(self.save_path, value_tensor)

        return value_tensor
            

        #     if isinstance(v, Box):
        #         low = self.sample_range[k][0]
        #         high = self.sample_range[k][1]
        #         ## sample all possible values
        #         # samples = np.random.uniform(low=low, high=high, size=(sample_size, v.shape[0])).astype(np.float32)
               
        #         # sample only int
        #         samples = np.arange(low, high+1, 1).astype(np.float32)
                
        #         ## sample with 0.5 interval
        #         # low_scaled = int(low * 2)
        #         # high_scaled = int(high * 2)
        #         # samples = np.random.randint(low=low_scaled, high=high_scaled, size=(sample_size, v.shape[0])).astype(np.float32) * 0.5
        #         sample_dict[k] = samples
        #     else:
        #         sample_dict[k] = np.random.choice([1, 0], size=sample_size).reshape(-1, 1).astype(np.int32)
        # print(self.sample_range)


        return    


    
    def xadd2tree(self, xadd, value_id_pe):

        # Generate samples
        samples = self.extract_samples(self.sample_size)

        predictions = np.zeros(self.sample_size, dtype=np.float32)

        var_set = xadd.collect_vars(value_id_pe)
        var_dict = {}
        for i in var_set:
            var_dict[f"{i}"] = i

        for i in range(self.sample_size):
            c_assign = {}
            b_assign = {}
            for k in samples.keys():
                v = samples[k][i][0]
                if isinstance(v, np.float32):
                    c_assign[var_dict[k]] = float(v)
                else:
                    b_assign[var_dict[k]] = bool(v)
            value = self.xadd_context_target.evaluate(value_id_pe, bool_assign=b_assign, cont_assign=c_assign)
            predictions[i] = value
        

        samples_df = {k:v.reshape(-1) for k,v in samples.items()}
        samples_df = pd.DataFrame(samples_df)[self.env.observation_list] # reorder columns accoring to observation_list

        tree_clf = DecisionTreeRegressor(max_depth=None)
        tree_clf = tree_clf.fit(samples_df, predictions)

        x = np.arange(0, 11, 1)
        y = np.arange(0, 11, 1)

        X, Y = np.meshgrid(x, y)

        Z = np.zeros_like(X, dtype=float)

        warnings.filterwarnings("ignore", category=FutureWarning)

        for i in range(len(x)):
            for j in range(len(y)):
                obs = pd.DataFrame({'pos_x___a1': np.array([i], dtype=np.float32), 'pos_y___a1': np.array([j], dtype=np.float32), 'has_mineral___a1':1})
                value = tree_clf.predict(obs)
                Z[i][j] = value

        np.set_printoptions(precision=2)

        print(Z.T)

        Z = np.zeros_like(X, dtype=float)

        warnings.filterwarnings("ignore", category=FutureWarning)

        for i in range(len(x)):
            for j in range(len(y)):
                c_assign = {var_dict['pos_x___a1']:i, var_dict['pos_y___a1']:j}
                b_assign = {var_dict['has_mineral___a1']:True}
                value = self.xadd_context_target.evaluate(value_id_pe, bool_assign=b_assign, cont_assign=c_assign)
                Z[i][j] = value
            
        print(Z.T)
    


        return tree_clf
            
        





        

    
    def get_xadd_model_from_file(self, f_domain, f_instance):
        # Read and parse domain and instance
        reader = RDDLReader(f_domain, f_instance)
        domain = reader.rddltxt
        parser = RDDLParser(None, False)
        parser.build()

        # Parse RDDL file
        rddl_ast = parser.parse(domain)

        # Ground domain
        grounder = RDDLGrounder(rddl_ast)
        model = grounder.Ground()

        # XADD compilation
        xadd_model = RDDLModelWXADD(model)
        xadd_model.compile(simulation=True)

        context = xadd_model._context
        return xadd_model, context

    
    def extract_samples(self, sample_size, sample_range=None):
        if sample_range == None:
            sample_range = self.sample_range
        sample_dict = {}
        for k, v in self.env.observation_space.items():
            if isinstance(v, Box):
                low = sample_range[k][0]
                high = sample_range[k][1]
                arange = np.arange(low, high+sample_range[k][2], sample_range[k][2])
                samples = np.random.choice(arange, size=sample_size).reshape(-1, 1).astype(np.float32)    
                sample_dict[k] = samples
            else:
                sample_dict[k] = np.random.choice([1, 0], size=sample_size).reshape(-1, 1).astype(np.int32)
        
        return sample_dict

    def network2tree(self, max_depth=None):
        """
        Convert the neural network to a decision tree classifier.
        """
        # Generate samples
        samples = self.extract_samples(self.sample_size)

        predictions, _ = self.network.predict(samples)

        samples_df = {k:v.reshape(-1) for k,v in samples.items()}
        samples_df = pd.DataFrame(samples_df)[self.env.observation_list] # reorder columns accoring to observation_list

        tree_clf = tree.DecisionTreeClassifier(max_depth=max_depth)
        tree_clf = tree_clf.fit(samples_df, predictions)
        return tree_clf


        # x = np.arange(0, 11, 1)
        # y = np.arange(0, 11, 1)

        # X, Y = np.meshgrid(x, y)


        # Z = np.zeros_like(X, dtype=float)

        # for i in range(len(x)):
        #     for j in range(len(y)):
        #         obs = {'pos_x___a1': np.array([i], dtype=np.float32), 'pos_y___a1': np.array([j], dtype=np.float32), 'has_mineral___a1':1}
        #         value, _ = self.network.predict(obs)
        #         Z[i][j] = value

        # print(Z.T)

        # Z = np.zeros_like(X, dtype=float)

        # for i in range(len(x)):
        #     for j in range(len(y)):
        #         obs = pd.DataFrame({'pos_x___a1': np.array([i], dtype=np.float32), 'pos_y___a1': np.array([j], dtype=np.float32), 'has_mineral___a1':1})
        #         value = clf.predict(obs)
        #         Z[i][j] = value
        #         # inputs.append(np.array([i,j]))
        #         # labels.append(env.action_list[value])

        # print(self.env.action_list)
        # print(Z.T)
    
    def tree2xaddpolicy(self, tree_clf):
        
        policy_dict = self.tree2dict(tree_clf, self.env.observation_list)

        xadd_str = self.policy_dict2xadd_str(policy_dict, self.env.action_list)

        policy_id = self.xadd_context_target.import_xadd(xadd_str=xadd_str, 
                                                locals=self.xadd_model_target.ns)
        policy_id = self.xadd_context_target.reduce_lp(policy_id)

        xadd_policy_dict = self.gen_policy_dict(self.env.action_list, policy_id, self.xadd_context_target)

        return xadd_policy_dict

    def gen_policy_dict(self, action_list, all_policy_id, xadd):
        policy_dict = {}
        for action in action_list:
            if action in xadd._str_var_to_var.keys():
                sub_dict = {}
                for a in action_list:
                    if a in xadd._str_var_to_var.keys():
                        if a == action:
                            sub_dict[xadd._str_var_to_var[a]] = 1
                        else:
                            sub_dict[xadd._str_var_to_var[a]] = 0

                policy_id = xadd.substitute(all_policy_id, sub_dict)
                policy_id = xadd.reduce_lp(policy_id)
                # policy_id = xadd.unary_op(policy_id, 'float')
                # policy_node = xadd._id_to_node[policy_id]
                # policy_node.turn_off_print_node_info()
                # policy_dict[action] = policy_node
                policy_dict[action] = policy_id
            else:
                policy_dict[action] = xadd.ZERO

        return policy_dict


    def policy_dict2xadd_str(self, node, action_list):
        # Base case: if the node is a leaf (i.e., a string), return it directly
        if not isinstance(node, dict):
            return f"([{action_list[node]}])"

        # Recursive case: process left and right children
        left_str = self.policy_dict2xadd_str(node['left'], action_list) if 'left' in node else ''
        right_str = self.policy_dict2xadd_str(node['right'], action_list) if 'right' in node else ''

        # Format the current node string

        var = node['feature']
        if isinstance(self.env.observation_space[var], Box):
            node_str = f"( [{node['feature']} <= {node['threshold']}] {left_str} {right_str} )"
        else:
            node_str = f"( [{node['feature']}] {left_str} {right_str} )"
        return node_str

    def tree2dict(self, clf, feature_names=None):
        tree = clf.tree_
        if feature_names is None:
            feature_names = range(clf.max_features_)
        
        # Build tree nodes
        tree_nodes = []
        for i in range(tree.node_count):
            if (tree.children_left[i] == tree.children_right[i]):
                tree_nodes.append(
                    clf.classes_[np.argmax(tree.value[i])]
                )
            else:
                tree_nodes.append({
                    "feature": feature_names[tree.feature[i]],
                    "threshold": tree.threshold[i],
                    "left": tree.children_left[i],
                    "right": tree.children_right[i],
                })
        
        # Link tree nodes
        for node in tree_nodes:
            if isinstance(node, dict):
                node["left"] = tree_nodes[node["left"]]
            if isinstance(node, dict):
                node["right"] = tree_nodes[node["right"]]
        
        # Return root node
        return tree_nodes[0]



        # fig = plt.figure(figsize=(50, 30))
        # tree.plot_tree(clf)
        # fig.savefig('test_tree.png')

        # x = np.arange(0, 11, 1)
        # y = np.arange(0, 11, 1)

        # X, Y = np.meshgrid(x, y)


        # Z = np.zeros_like(X, dtype=float)

        # for i in range(len(x)):
        #     for j in range(len(y)):
        #         obs = {'pos_x___a1': np.array([i], dtype=np.float32), 'pos_y___a1': np.array([j], dtype=np.float32), 'has_mineral___a1':1}
        #         value, _ = self.network.predict(obs)
        #         Z[i][j] = value

        # print(Z.T)

        # Z = np.zeros_like(X, dtype=float)

        # for i in range(len(x)):
        #     for j in range(len(y)):
        #         obs = pd.DataFrame({'pos_x___a1': np.array([i], dtype=np.float32), 'pos_y___a1': np.array([j], dtype=np.float32), 'has_mineral___a1':1})
        #         value = clf.predict(obs)
        #         Z[i][j] = value
        #         # inputs.append(np.array([i,j]))
        #         # labels.append(env.action_list[value])

        # print(self.env.action_list)
        # print(Z.T)

        
        
        
        

    def convert_to_xadd(self):
        """
        Convert the decision tree to an XADD.
        """
        pass

    def predict(self, input_data):
        """
        Make predictions using the XADD model.
        """
        passse