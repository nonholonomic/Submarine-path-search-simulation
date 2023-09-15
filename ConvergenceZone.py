import math


class CZ:
    def __init__(self, setting):
        self.centralRadius = setting.CONVERGENCE_ZONE_CENTRAL_RADIUS
        self.halfWidth = setting.CONVERGENCE_ZONE_HALF_WIDTH
        self.speedError = setting.CONVERGENCE_ZONE_SPEED_ERROR
        self.courseError = math.radians(setting.CONVERGENCE_ZONE_COURSE_ERROR)
        self.bearingError = math.radians(setting.CONVERGENCE_ZONE_BEARING_ERROR)
        self.rangeError = setting.CONVERGENCE_ZONE_RANGE_ERROR
        self.innerCZR = self.centralRadius - self.halfWidth
        self.outerCZR = self.centralRadius + self.halfWidth
