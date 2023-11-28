import torch
import torch.nn as nn

from gymnasium.spaces import Discrete, Dict, Box

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from matplotlib import pyplot as plt

from pyRDDLGym.Core.Grounder.RDDLGrounder import RDDLGrounder
from pyRDDLGym.Core.Parser.parser import RDDLParser
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD

class ValueGenerator:
    def __init__(self, env, domain_path, instance_path, network, sample_range, max_depth=None):
        self.env = env
        self.network = network
        self.sample_range = sample_range
        self.max_depth = max_depth
        self.xadd_model = self.load_xadd(domain_path, instance_path)
        self.xadd_context = self.xadd_model._context
        self.decision_tree_nn = self.network2tree(max_depth=max_depth)
        self.decision_tree_xadd = None
        self.policy_xadd = self.tree2xaddpolicy()
        
    
    def load_xadd(self, domain, instance):
        reader = RDDLReader(domain, instance)
        domain = reader.rddltxt
        rddl_parser = RDDLParser(None, False)
        rddl_parser.build()

        # Parse RDDL file.
        rddl_ast = rddl_parser.parse(domain)

        # Ground domain.
        grounder = RDDLGrounder(rddl_ast)
        model = grounder.Ground()

        # XADD compilation.
        xadd_model = RDDLModelWXADD(model, simulation=False)
        xadd_model.compile()

        return xadd_model


    
    def extract_samples(self, sample_size = 100000, sample_range=None):
        if sample_range == None:
            sample_range = self.sample_range
        sample_dict = {}
        for k, v in self.env.observation_space.items():
            if isinstance(v, Box):
                low = sample_range[k][0]
                high = sample_range[k][1]
                samples = np.random.uniform(low=low, high=high, size=(sample_size, v.shape[0])).astype(np.float32)
                sample_dict[k] = samples
            else:
                sample_dict[k] = np.random.choice([1, 0], size=sample_size).reshape(-1, 1).astype(np.int32)
        
        return sample_dict

    def network2tree(self, max_depth=None):
        """
        Convert the neural network to a decision tree classifier.
        """
        # Generate samples
        samples = self.extract_samples()
        predictions, _ = self.network.predict(samples)

        samples_df = {k:v.reshape(-1) for k,v in samples.items()}
        samples_df = pd.DataFrame(samples_df)[self.env.observation_list] # reorder columns accoring to observation_list

        tree_clf = tree.DecisionTreeClassifier(max_depth=max_depth)
        tree_clf = tree_clf.fit(samples_df, predictions)
        return tree_clf
    
    def tree2xaddpolicy(self):
        
        policy_dict = self.tree2dict(self.decision_tree_nn, self.env.observation_list)

        xadd_str = self.policy_dict2xadd_str(policy_dict, self.env.action_list)

        policy_id = self.xadd_context.import_xadd(xadd_str=xadd_str, 
                                                locals=self.xadd_context._str_var_to_var)
        policy_id = self.xadd_context.reduce_lp(policy_id)

        xadd_policy_dict = self.gen_policy_dict(self.env.action_list, policy_id, self.xadd_context)

        for k, v in xadd_policy_dict.items():
            print(k)
            print(v)

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
                policy_id = xadd.unary_op(policy_id, 'float')
                policy_node = xadd._id_to_node[policy_id]
                policy_node.turn_off_print_node_info()
                policy_dict[action] = str(policy_node)
                policy_node.turn_on_print_node_info()
            else:
                policy_dict[action] = "([0])"
            
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
        pass