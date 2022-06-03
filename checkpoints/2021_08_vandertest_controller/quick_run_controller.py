"""Quick run."""
import numpy as np
import onnx  # pip install onnx
import onnxruntime as ort  # pip install onnxruntime
import matplotlib.pyplot as plt

CONTROLLER_PATH = "./vandertest_controller.onnx"

# load RL controller
model = onnx.load_model(CONTROLLER_PATH)
ort_session = ort.InferenceSession(CONTROLLER_PATH)


def get_accel(state):
    """
    Get requested acceleration.

    state is [av speed, leader speed, headway] (no normalization needed)
    output is instant acceleration to apply to the AV
    """
    data = np.array([state]).astype(np.float32)
    outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: data})
    return outputs[0][0][0]


# initialize AV
av_positions = [0]
av_speeds = [15]

# initialize leader
leader_positions = [50]
leader_speeds = [15]

# run AV behind leader
dt = 0.1
times = [0.0]

for _ in range(1000):
    av_pos = av_positions[-1]
    av_speed = av_speeds[-1]
    leader_pos = leader_positions[-1]
    leader_speed = leader_speeds[-1]

    # get AV accel
    av_space_gap = leader_pos - av_pos
    av_accel = get_accel([av_speed, leader_speed, av_space_gap])

    # update AV
    new_av_speed = av_speed + dt * av_accel
    new_av_pos = av_pos + dt * new_av_speed
    av_speeds.append(new_av_speed)
    av_positions.append(new_av_pos)

    # update leader
    new_leader_pos = leader_pos + dt * leader_speed
    leader_speeds.append(leader_speed)  # constant leader speed
    leader_positions.append(new_leader_pos)

    times.append(times[-1] + dt)

plt.figure()
plt.plot(times, av_positions, label='AV')
plt.plot(times, leader_positions, label='leader')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.show()
