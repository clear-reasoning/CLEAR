import argparse
import json
import os

import onnx
from onnx_tf.backend import prepare
import torch
from torch import nn

from env.trajectory_env import TrajectoryEnv
parser = argparse.ArgumentParser()
parser.add_argument('--cp_path', type=str, help='Path to the checkpoint you want to convert.'
                                                'This should include the .zip file that you want opened')
parser.add_argument('--logdir', type=str, help='Where to save the final model')

args = parser.parse_args()

# grab the configs we need to load the model
config_path = os.path.join('/'.join(args.cp_path.split('/')[:-2]) ,'configs.json')
with open(config_path, 'r') as fp:
    config = json.load(fp)
env = TrajectoryEnv(config['env_config'])
# grab the algo
algo = None
if config['algorithm'] == '<class \'algos.ppo.ppo.PPO\'>':

    from stable_baselines3 import PPO
    model = PPO.load(args.cp_path, env=env)


    # now we need to pull the model out of stable baselines so that we can actually use it
    class Policy(nn.Module):
        def __init__(self, baselines_model):
            super(Policy, self).__init__()

            self.mlp_extractor = baselines_model.policy.policy_extractor
            self.action_net = baselines_model.policy.action_net

        def forward(self, x):
            x = self.mlp_extractor(x)
            return self.action_net(x)


    policy = Policy(model)
elif config['algorithm'] == '<class \'algos.td3.td3.TD3\'>':
    from stable_baselines3 import TD3
    model = TD3.load(args.cp_path, env=env)

x = torch.Tensor([[0, 0, 0]])
torch_out = policy(x)

if args.logdir:
  output_logdir = args.logdir
else:
  output_logdir = os.path.dirname(args.cp_path)

onnx_path = os.path.join(output_logdir, "super_resolution.onnx")

# # Export the model
torch.onnx.export(policy,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  onnx_path,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

onnx_model = onnx.load(onnx_path)
tf_rep = prepare(onnx_model)
tf_path = os.path.join(output_logdir, "model.pb")
tf_rep.export_graph(tf_path)