from trajectory.env.trajectory_env import TrajectoryEnv, DEFAULT_ENV_CONFIG
from pathlib import Path
import json
from trajectory.algos.ppo.ppo import PPO as AugmentedPPO
from trajectory.visualize.time_space_diagram import plot_time_space_diagram
import pprint
import matplotlib.pyplot as plt
import copy
import matplotlib
matplotlib.use('TkAgg')

CP_ACCEL_PATH = "local/itsc/itsc_nl_2_21h19m54s/checkpoints/2000.zip"

cp_path = Path(CP_ACCEL_PATH)
log_path = cp_path.parent.parent / "logs"
log_path.mkdir(exist_ok=True)
config_path = cp_path.parent.parent / "configs.json"
with open(config_path, 'r') as f:
    config = json.load(f)
base_env_config = DEFAULT_ENV_CONFIG
base_env_config.update(config["env_config"])
pprint.pprint(base_env_config)

model = AugmentedPPO.load(str(cp_path))
model_fn = lambda state: model.predict(state, deterministic=True)[0]

if True:
    env_config = copy.deepcopy(base_env_config)
    # modify env config
    env_config["fixed_traj_path"] = "dataset/data_v2_preprocessed_west/2021-04-22-12-47-13_2T3MWRFVXLW056972_masterArray_0_7050/trajectory.csv"
    # env_config["fixed_traj_path"] = "dataset/data_v2_preprocessed_west/2021-03-09-13-35-04_2T3MWRFVXLW056972_masterArray_0_6825/trajectory.csv"
    env_config["whole_trajectory"] = True
    # env_config["av_controller"] = "rl"
    env_config["platoon"] = "av human*20"
    # env_config["platoon"] = "(av human*24)*20"

    # create env
    env = TrajectoryEnv(config=env_config, _simulate=True, _verbose=True, _model_fn=model_fn)
    env.do_not_reset_on_end_of_horizon = True  # dont reset at the end of eval rollout otherwise data gets erased
    env.reset()

    # step through the whole trajectory
    while not env.end_of_horizon:
        _, _, done, _ = env.step(None)

    # # dict_keys(['0_trajectory_leader', '1_rl_av', '2_idm_human', '3_idm_human', '4_idm_human', '5_idm_human', '6_idm_human'])
    # print(env.sim.data_by_vehicle.keys())
    # # dict_keys(['time', 'step', 'id', 'position', 'speed', 'accel', 'headway', 'leader_speed', 'speed_difference', 'time_gap', 'time_to_collision', 'leader_id', 'follower_id', 'road_grade', 'altitude', 'instant_energy_consumption', 'total_energy_consumption', 'total_distance_traveled', 'total_miles', 'total_gallons', 'avg_mpg', 'realized_accel', 'target_accel_no_noise_no_failsafe', 'target_accel_with_noise_no_failsafe', 'target_accel_no_noise_with_failsafe', 'vdes'])
    # print(env.sim.data_by_vehicle['1_rl_av'].keys())
    keys = list(env.sim.data_by_vehicle.keys())
    assert 'trajectory_leader' in keys[0]
    assert 'av' in keys[1]
    assert 'idm' in keys[21]
    t = env.sim.data_by_vehicle[keys[0]]['time']
    v_leader = env.sim.data_by_vehicle[keys[0]]['speed']
    v_av = env.sim.data_by_vehicle[keys[1]]['speed']
    v_idm = env.sim.data_by_vehicle[keys[21]]['speed']
    plt.figure()
    plt.plot(t, v_leader, label='leader')
    plt.plot(t, v_av, label='av')
    plt.plot(t, v_idm, label='idm')
    plt.legend()
    plt.show()

    # emissions_path = log_path / "emissions.csv"
    # env.gen_emissions(emissions_path=str(emissions_path), upload_to_leaderboard=False)

    # tsd_path = log_path / "tsd.png"
    # plot_time_space_diagram(str(emissions_path), save_path=str(tsd_path))
    # print(tsd_path)
