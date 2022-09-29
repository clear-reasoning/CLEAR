/**
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
{
    // build state
    std::vector<float> state = {
        this_vel / 40.0f,
        target_speed / 40.0f,
        static_cast<float>(max_headway) / 1.0f,
        mega.get_speed_setting() / 40.0f,
        mega.get_gap_setting() / 3.0f,
    };

    // get accel from neural network
    auto[speed_action, gap_action] = onnx_checkpoint.forward(state);
    
    // clip actions
    speed_action = std::clamp(speed_action, -1.0f, 1.0f);
    gap_action = std::clamp(gap_action, -1.0f, 1.0f);

    // compute actions for ACC
    int speed_setting = static_cast<int>((speed_action + 1.0f) * 20.0f);
    int gap_setting = gap_action > (1.0f / 3.0f) ? 1 : gap_action > (-1.0f / 3.0f) ? 2 : 3;

    // compute accel
    const float accel = mega.get_accel(speed_setting, gap_setting);

    return accel;
}
