import numpy as np

class IDMController(object):
    def __init__(self, v0=30, T=1, a=1.3, b=2.0, delta=4, s0=2):
        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.delta = delta
        self.s0 = s0

    def get_accel(self, this_vel, lead_vel, headway):
        """See parent class."""
        # in order to deal with ZeroDivisionError
        if abs(headway) < 1e-3:
            headway = 1e-3

        if lead_vel is None:  # no car ahead
            s_star = 0
        else:
            s_star = self.s0 + max(
                0, this_vel * self.T + this_vel * (this_vel - lead_vel) /
                (2 * np.sqrt(self.a * self.b)))

        return self.a * (1 - (this_vel / self.v0)**self.delta - (s_star / headway)**2)

