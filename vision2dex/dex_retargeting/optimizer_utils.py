import numpy as np

class LPFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.y = None
        self.is_init = False

    def next(self, x):
        if not self.is_init:
            self.y = x
            self.is_init = True
            return self.y.copy()
        self.y = self.y + self.alpha * (x - self.y)
        return self.y.copy()

    def reset(self):
        self.y = None
        self.is_init = False

    # --- convenience helpers ---
    @staticmethod
    def alpha_from_tau(dt: float, tau: float) -> float:
        """
        Choose alpha from a desired time constant tau (seconds):
            alpha = dt / (tau + dt)
        """
        if dt <= 0 or tau <= 0:
            raise ValueError("dt and tau must be positive.")
        return dt / (tau + dt)

    @staticmethod
    def alpha_from_cutoff(dt: float, fc_hz: float) -> float:
        """
        Choose alpha from a desired -3 dB cutoff frequency fc (Hz):
            tau = 1 / (2*pi*fc),  alpha = dt / (tau + dt)
        """
        if dt <= 0 or fc_hz <= 0:
            raise ValueError("dt and fc_hz must be positive.")
        tau = 1.0 / (2.0 * np.pi * fc_hz)
        return dt / (tau + dt)
