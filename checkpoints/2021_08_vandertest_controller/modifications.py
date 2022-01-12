import torch


def forward(self, in_vector):  # in_vector (non-normalized) : [av_speed, leader_speed, headway]
    # normalize input
    x = torch.div(in_vector, self.normalization_tensor)

    # get and bound RL accel
    x = self.mlp_extractor(x)
    rl_action = torch.clamp(self.action_net(x), min=-3.0, max=1.5)

    # extract state variables
    av_speed, _ = torch.div(in_vector, self.av_speed_filter_tensor).max(dim=1)
    leader_speed, _ = torch.div(in_vector, self.leader_speed_filter_tensor).max(dim=1)
    headway, _ = torch.div(in_vector, self.headway_filter_tensor).max(dim=1)

    time_headway = headway / av_speed
    speed_diff = av_speed - leader_speed
    # we don't ever want to divide by zero and we don't care about negative values so we just max
    # with 0.1
    speed_diff, _ = torch.cat((torch.Tensor([0.1]), speed_diff)).max(dim=0)
    # 1 if speed diff greater than 0.5
    max_decel = -3
    # 1 if speed diff greater than 0.5
    TTC = headway / speed_diff
    small_TTC = (1 / (1.0 + torch.exp(-1.5 * (6.5 - TTC))))

    # now above 120 we do a logistic decay from the current accel to 0.5
    final_large_gap_accel = 0.75
    is_large_gap = (1 / (1.0 + torch.exp(-0.1 * (headway - 120.0))))
    modified_action = rl_action * (1 - is_large_gap) + final_large_gap_accel * is_large_gap

    modified_action = (1 - small_TTC) * modified_action + small_TTC * max_decel

    # rebound accel
    modified_action = torch.clamp(modified_action, min=-3.0, max=1.5)

    return modified_action
