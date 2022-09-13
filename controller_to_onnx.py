import argparse
import json
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
import torch

from trajectory.algos.ppo.ppo import PPO as AugmentedPPO
from trajectory.env.trajectory_env import TrajectoryEnv


# https://stable-baselines3.readthedocs.io/en/master/guide/export.html
class OnnxablePolicy(torch.nn.Module):
    def __init__(self, model):
        super(OnnxablePolicy, self).__init__()
        model.policy.to('cpu')
        self.mlp_extractor = model.policy.policy_extractor
        self.action_net = model.policy.action_net

    def forward(self, observation):
        x = self.mlp_extractor(observation)
        action = self.action_net(x)
        return action

MODEL_INPUT_NAME = 'onnx::Gemm_0'


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to the controller directory. '
                        'Must contain a checkpoint.zip file and a configs.json file')
    args = parser.parse_args()
    path = Path(args.path)

    # load config
    config_path = path / 'configs.json'
    print(f'> Loading config from {config_path}')
    with open(config_path, 'r') as fp:
        config = json.load(fp)
    print(f'> Config: {config}')

    # load model
    model_path = path / 'checkpoint.zip'
    print(f'\n> Loading model from {model_path}')
    model = AugmentedPPO.load(model_path)
    n_observations = model.observation_space.shape[0] // 2
    print(f'> Model has {n_observations} inputs, input name is {MODEL_INPUT_NAME}')
    onnxable_model = OnnxablePolicy(model)

    # export model to onnx
    onnx_path = path / 'model.onnx'
    dummy_input = torch.randn(1, n_observations)
    torch.onnx.export(onnxable_model, dummy_input, str(onnx_path), opset_version=9)
    print(f'> Wrote ONNX model at {onnx_path}\n')

    # load onnx model and test it
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    ort_sess = ort.InferenceSession(str(onnx_path))
    for i in range(3):
        observation = np.round(np.random.rand(1, n_observations).astype(np.float32), 3)
        onnx_action = ort_sess.run(None, {MODEL_INPUT_NAME: observation})
        model_action = model.predict(np.concatenate((observation, observation), axis=1), deterministic=True)[0]
        onnx_action_clipped = np.clip(onnx_action, model.action_space.low, model.action_space.high)
        assert np.all((model_action[0] - onnx_action_clipped[0][0]) < 1e-5)
        print(f'> Test input #{i+1}: {observation}')
        print(f'> Expected output #{i+1}: {np.round(onnx_action, 5)}')
    print(f'\n> Note: output should be clipped between {model.action_space.low} and {model.action_space.high}\n')

    # load environment
    env = TrajectoryEnv(config['env_config'], _verbose=False)

    # get state info
    sample_state = env.get_base_state()
    state_keys = list(sample_state.keys())
    state_values = [x[0] for x in sample_state.values()]
    state_normalizations = [x[1] for x in sample_state.values()]
    print(f'> Base state is: {state_keys}')
    print(f'  with normalization: {state_normalizations}')
    print(f'  eg. (without norm): {state_values}')

    state_to_cpp_map = {
        'speed': 'this_vel',
        'leader_speed': 'lead_vel',
        'headway': 'headway',
        **{f'leader_speed_{i}': f'prev_vels[{i-1}]' for i in range(1000)},
        'target_speed': 'target_speed',
        'max_headway': 'static_cast<float>(max_headway)',
        'speed_setting': 'mega.get_speed_setting()',
        'gap_setting': 'mega.get_gap_setting()',
    }

    states_cpp_str = '\n        '.join([f'{state_to_cpp_map[state_key]} / {state_norm}f,'
                                        for state_key, state_norm in zip(state_keys, state_normalizations)])

    cpp_pseudocode = f'''/**
 * Get RL controller acceleration.
 *
 * @param this_vel: AV velocity in m/s
 * @param lead_vel: leader velocity in m/s
 * @param headway: AV gap in m
 * @param prev_vels: vector of past leader velocities in m/s (where prev_vels[0] is the leader speed at t-1)
 * @param mega: MegaController object
 * @param target_speed: speed planner target speed in m/s
 * @param max_headway: speed planner gap flag (boolean)
 
 * @return AV acceleration in m/s/s
 */
float get_accel(float this_vel, float lead_vel, float headway, std::vector<float> prev_vels,
                float target_speed, bool max_headway,
                MegaController& mega, Model onnx_checkpoint, float sim_step=0.1)
{{
    // build state
    std::vector<float> state = {{
        {states_cpp_str}
    }};

    // get accel from neural network
    auto[speed_action, gap_action] = onnx_checkpoint.forward(state);

    // clip actions
    speed_action = std::clamp(speed_action, -1.0f, 1.0f);
    gap_action = std::clamp(gap_action, -1.0f, 1.0f);
    
    // compute actions for ACC
    int speed_setting = static_cast<int>((speed_action + 1.0f) * 20.0f);
    int gap_setting = gap_action > (1.0f / 3.0f) ? 1 : gap_action > (-1.0f / 3.0f) ? 2 : 3;

    // apply gap closing and failsafe
    const float gap_closing_threshold = std::max({int(env.max_headway)}.0f, {int(env.max_time_headway)}.0f * ego_vel);
    const float failsafe_threshold = 6.0f * ((ego_vel + 1.0f + ego_vel * 4.0f / 30.0f) - lead_vel);
    
    if (headway > gap_closing_threshold) {{
        speed_setting = 40;
    }}
    elif (headway < failsafe_threshold) {{
        speed_setting = 0;
    }}
    
    // compute accel
    const float accel = mega.get_accel(speed_setting, gap_setting);

    return accel;
}}'''
    print(f'\n> C++ pseudocode:\n{cpp_pseudocode}')
