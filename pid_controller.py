class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.error_p_old = 0
        self.error_i = 0
 
    def update(self, dt, setpoint, actual):
        if setpoint is not None:
            self.setpoint = setpoint
        error = self.setpoint - actual
        error_p = error
        self.error_i += error_p - self.error_p_old
        error_d = (error_p - self.error_p_old) / dt
        self.error_p_old = error_p
        return self.kp * error_p + self.ki * self.error_i + self.kd * error_d