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
    
    # conversion for discrete actions
    env_config = config['env_config']
    discrete_actions = env_config.get('acc_output', False) == True and env_config.get('acc_continuous') == False
    if discrete_actions:
        acc_num_speed_settings = int((env_config['acc_max_speed'] - env_config['acc_min_speed'])/ env_config['acc_speed_step'] + 1)
        gap_action_set = np.array([1, 2, 3])
        speed_action_set = np.arange(env_config['acc_min_speed'],
                                     env_config['acc_max_speed'] + env_config['acc_speed_step'],
                                     env_config['acc_speed_step'])

    # load onnx model and test it
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    ort_sess = ort.InferenceSession(str(onnx_path))
    for i in range(3):
        observation = np.round(np.random.rand(1, n_observations).astype(np.float32), 3)
        onnx_action = ort_sess.run(None, {MODEL_INPUT_NAME: observation})
        model_action = model.predict(np.concatenate((observation, observation), axis=1), deterministic=True)[0]
        if discrete_actions:
            onnx_action_clipped = [[np.argmax(onnx_action[0][0][0:61]), np.argmax(onnx_action[0][0][61:64])]]
        else:
            onnx_action_clipped = np.clip(onnx_action, model.action_space.low, model.action_space.high)
        assert np.all((model_action[0] - onnx_action_clipped[0][0]) < 1e-5)
        print(f'> Test input #{i+1}: {observation}')
        print(f'> Expected output #{i+1}: {np.round(onnx_action, 5)}')
    if discrete_actions:
        print(f'\n> Note: output is discrete logits, first {acc_num_speed_settings} are for speed settings, next 3 for gap settings')
    else:
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
        'gap_closing': f"std::max({int(env_config['max_headway'])}.0f, {int(env_config['max_time_headway'])}.0f * this_vel)",
        'failsafe': f"6.0f * ((this_vel + 1.0f + this_vel * 4.0f / 30.0f) - lead_vel)",
    }

    states_cpp_str = '\n        '.join([f'{state_to_cpp_map[state_key]} / {state_norm}f,'
                                        for state_key, state_norm in zip(state_keys, state_normalizations)])

    if discrete_actions:
        action_computation_str = f'''// get logits from neural network
    std::vector<float> logits = onnx_checkpoint.forward(state);
    
    // find argmax of speed setting logits (indexes 0 to {acc_num_speed_settings} excluded)
    int speed_action = 0;
    float max_speed_logit = logits[0];
    for (int i = 1; i < {acc_num_speed_settings}; ++i) {{
        if (logits[i] > max_speed_logit) {{
            speed_action = i;
            max_speed_logit = logits[i];
        }}
    }}
    
    // find argmax of gap setting logits (indexes {acc_num_speed_settings} to {acc_num_speed_settings + 3} excluded)
    int gap_action = 0;
    float max_gap_logit = logits[0];
    for (int i = {acc_num_speed_settings}; i < {acc_num_speed_settings + 3}; ++i) {{
        if (logits[i] > max_gap_logit) {{
            gap_action = i;
            max_gap_logit = logits[i];
        }}
    }}
    
    // convert discrete actions to respective settings
    float speed_setting = {env_config['acc_min_speed']} + {env_config['acc_speed_step']} * i;
    int gap_setting = gap_action + 1;'''
    else:  
        action_computation_str = f'''// get accel from neural network
    auto[speed_action, gap_action] = onnx_checkpoint.forward(state);
    
    // clip actions
    speed_action = std::clamp(speed_action, -1.0f, 1.0f);
    gap_action = std::clamp(gap_action, -1.0f, 1.0f);

    // compute actions for ACC
    int speed_setting = static_cast<int>((speed_action + 1.0f) * 20.0f);
    int gap_setting = gap_action > (1.0f / 3.0f) ? 1 : gap_action > (-1.0f / 3.0f) ? 2 : 3;'''

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

    {action_computation_str}

    // compute accel
    const float accel = mega.get_accel(speed_setting, gap_setting);

    return accel;
}}'''
    print(f'\n> C++ pseudocode:\n{cpp_pseudocode}')
