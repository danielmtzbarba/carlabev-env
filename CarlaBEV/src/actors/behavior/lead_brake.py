class LeadBrakeBehavior:
    def __init__(self, start_brake_t=3.5, dec_rate=1.0):
        self.start_brake_t = start_brake_t
        self.dec_rate = dec_rate
        self.braking = False

    def reset(self, actor):
        self.braking = False

    def apply(self, actor, t, dt):
        if t >= self.start_brake_t:
            self.braking = True

        if self.braking:
            actor.target_speed = max(0.0, actor.target_speed - self.dec_rate * dt)
