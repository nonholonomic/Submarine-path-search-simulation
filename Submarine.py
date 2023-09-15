import math
import scipy
import ConvergenceZone
import numpy as np


class Submarine:
    def __init__(self, env, run_setting, is_attacker, env_setting):
        # common setting
        self.env = env
        self.currentTime = env.now
        self.ambientNoiseLevel = run_setting.AMBIENT_NOISE_LEVEL
        self.STDOfPropagationLoss = run_setting.STD_OF_PROPAGATION_LOSS
        self.propagationLossCurve = env_setting.PROPAGATION_LOSS
        self.rangeCurve = env_setting.RANGE_IN_MILES
        self.randomSeed = run_setting.RANDOM_SEED
        if is_attacker:
            self.trialProbEst = 0  # prob aggregation for this trial
            self.isInTransit = False
            self.isOnStation = False
            self.receivedIntel = False
            self.position = np.asarray([0.0, 0.0])
            self.course = math.radians(run_setting.ATTACKER_INITIAL_COURSE)
            self.speed = run_setting.ATTACKER_INITIAL_SPEED
            velocity_x = self.speed * math.cos(self.course)
            velocity_y = self.speed * math.sin(self.course)
            self.velocity = np.asarray([velocity_x, velocity_y])
            self.transitSpeed = run_setting.TRANSIT_SPEED
            self.searchSpeed = run_setting.SEARCH_SPEED
            self.searchLegTime = run_setting.SEARCH_LEG_TIME
            self.sonarGain = run_setting.ATTACKER_SONAR_GAIN
            self.czIntel = run_setting.CONVERGENCE_ZONE_INTELLIGENCE
            self.radiatedNoiseCurve = env_setting.ATTACKER_RADIATED_NOISE
            self.speedCurve = env_setting.ATTACKER_SPEED
            self.selfNoiseCurve = env_setting.ATTACKER_SELF_NOISE
            self.timeToCPA = -1
            self.CPA_position = np.asarray([0.0, 0.0])
            self.CPARange = 0
            self.timeToCZ = -1              # -1 denotes that the attacker will not enter the CZ of the target
            self.timeCZDetection = 0
            self.detectionProbDuringSearch = []
            self.estTargetCourse = None     # this is determined from intelligence. It must receive intel first.
            self.estTargetSpeed = None
            self.estTargetPos = None
            self.interceptPoint = None
            self.optimalInterceptions = {}      # this is for run summary, with perfect information
            self.realTargetCourse = None        # this is for run summary, with perfect information
            self.realTargetSpeed = None         # this is for run summary, with perfect information
            self.realTargetPos = None           # this is for run summary, with perfect information
            self.timeToInterception = 0
            self.interceptTimeUpperBound = run_setting.NOMINAL_TIME_TO_END_THE_TRIAL
            self.timeWhenReceivingIntel = 0
            self.timeOfIntel = 0
            # below for summarising detection data
            self.transitProbTimes = {}      # key: established prob time when inTransit; value: prob
            self.searchProbTimes = {}       # key: established prob time when onStation; value: prob
            self.occurrenceTransitMaxProb = 0
            self.occurrenceSearchMaxProb = 0
            self.occurrenceEvasionMaxProb = 0
            self.maxTransitProb = 0
            self.maxSearchProb = 0
            self.maxDetectionProb = max(self.maxTransitProb, self.maxSearchProb)
            self.timeOfMaxDetectionProb = 0             # time when having max prob
            self.broadcastTimeMaxProb = 0     # time when receiving the intel
            self.acquisitionTimeToMaxProb = 0           # time of the intel
            self.delaysAcquisitionToBroadcast = self.broadcastTimeMaxProb - self.acquisitionTimeToMaxProb
            self.delaysBroadcastToMaxDetectionProb = self.timeOfMaxDetectionProb - self.broadcastTimeMaxProb
        else:       # target setting
            self.isEvading = False
            self.trackNUm = run_setting.TARGET_TRACK_NUM
            self.legID = 0  # this will be updated when entering next leg, it is less than trackNUm
            self.position = np.asarray([run_setting.X_TARGET_INIT, run_setting.Y_TARGET_INIT])
            self.courseTable = np.deg2rad(run_setting.TARGET_BASE_COURSE)
            self.speedTable = run_setting.TARGET_BASE_SPEED
            self.course = self.courseTable[self.legID]
            self.speed = self.speedTable[self.legID]
            velocity_x = self.speed * math.cos(self.course)
            velocity_y = self.speed * math.sin(self.course)
            self.velocity = np.asarray([velocity_x, velocity_y])
            self.sonarGain = run_setting.TARGET_SONAR_GAIN
            self.radiatedNoiseCurve = env_setting.TARGET_RADIATED_NOISE
            self.speedCurve = env_setting.TARGET_SPEED
            self.selfNoiseCurve = env_setting.TARGET_SELF_NOISE
            self.counterdetectionTime = -1      # -1 denotes that the target will not be detected by the attacker
            self.counterdetectionRange = 0
            self.approachingAttacker = True
            self.wasApproachingAttacker = self.approachingAttacker
            # ConvergenceZone associates with a target
            self.CZ = ConvergenceZone.CZ(run_setting)
            # Target evasion has assigned rules of course and speed and reset to its base course and speed
            self.baseCourse = self.course
            self.baseSpeed = self.speed
            self.evasionOption = run_setting.EVASION_OPTION
            self.evasionTime = run_setting.EVASION_TIME
            self.evasionAngleIncrement = math.radians(run_setting.TARGET_EVASION_ANGLE)
            self.evasionSpeedIncrement = run_setting.EVASION_SPEED_INCREMENT
            self.customizedEvasionRoutine = None

    def compute_CPA_and_range(self, target):
        relative_track = self.position - target.position
        dist = np.linalg.norm(relative_track)
        unit_relative_track = relative_track / dist
        unit_target_velocity = target.velocity / target.speed
        dot_product = np.dot(unit_relative_track, unit_target_velocity)
        angle = np.arccos(dot_product)
        self.CPARange = abs(dist * math.sin(angle))
        # compute CPA point
        CPA_target_dist = abs(dist * math.cos(angle))
        CPA_x = CPA_target_dist * math.cos(target.course) + target.position[0]
        CPA_y = CPA_target_dist * math.sin(target.course) + target.position[1]
        self.CPA_position[0] = CPA_x
        self.CPA_position[1] = CPA_y

    def compute_time_to_CPA(self, target):  # refer to CPAT
        relative_track = self.position - target.position
        dist = np.linalg.norm(relative_track)
        unit_relative_track = relative_track / dist
        relative_speed_on_track = target.velocity.dot(unit_relative_track) - self.velocity.dot(unit_relative_track)
        if relative_speed_on_track <= 0:             # the range is increasing
            self.timeToCPA = -1
            target.wasApproachingAttacker = target.approachingAttacker
            target.approachingAttacker = False
        else:                               # the range is decreasing
            target.wasApproachingAttacker = target.approachingAttacker
            target.approachingAttacker = True
            # compute CPA time and CPA range
            self.compute_CPA_and_range(target)
            self.timeToCPA = self.CPARange / self.speed     # unit in hours

    def lookup_noise_curve(self):
        radiated_curve = scipy.interpolate.interp1d(self.speedCurve, self.radiatedNoiseCurve, fill_value='extrapolate')
        self_curve = scipy.interpolate.interp1d(self.speedCurve, self.selfNoiseCurve, fill_value='extrapolate')
        radiated_noise = radiated_curve(self.speed)
        self_noise = self_curve(self.speed)
        return radiated_noise, self_noise

    def compute_relative_speed(self, another):
        another_unit_v = another.velocity / np.linalg.norm(another.velocity)
        self_unit_v = self.velocity / np.linalg.norm(self.velocity)
        dot_product = np.dot(another_unit_v, self_unit_v)
        angle = np.arccos(dot_product)
        relative_speed = self.speed - another.speed * math.cos(angle)
        return relative_speed

    def compute_counterdetection_time(self, attacker):    # self is the target, refer to FIND
        # compute counterdetection range
        radiated_noise, self_noise = self.lookup_noise_curve()
        figure_of_merit = radiated_noise - max(self_noise, self.ambientNoiseLevel) + self.sonarGain
        prop_loss_curve = scipy.interpolate.interp1d(self.propagationLossCurve, self.rangeCurve,
                                                     fill_value='extrapolate')
        self.counterdetectionRange = prop_loss_curve(figure_of_merit)
        relative_track = attacker.position - self.position
        present_range = np.linalg.norm(relative_track)
        if present_range > self.counterdetectionRange:
            if not self.approachingAttacker:
                self.counterdetectionTime = -1
            elif attacker.CPARange > self.counterdetectionRange:
                self.counterdetectionTime = -1
            else:
                theta = math.acos(attacker.CPARange / self.counterdetectionRange)
                range_to_CPA = np.linalg.norm(attacker.CPA_position - self.position)
                range_to_counterdetection = range_to_CPA - self.counterdetectionRange * math.sin(theta)
                # compute relative speed on target course
                relative_speed = self.compute_relative_speed(attacker)
                self.counterdetectionTime = range_to_counterdetection / relative_speed
        else:
            self.counterdetectionTime = 0

    def compute_time_to_CZ(self, target):
        relative_track = target.position - self.position
        present_range = np.linalg.norm(relative_track)
        if target.approachingAttacker:
            if present_range > target.CZ.centralRadius:
                i_run = 3
            elif present_range > target.CZ.innerCZR:
                self.timeToCZ = 0
                return
            else:
                i_run = 2
        else:
            if present_range > target.CZ.outerCZR:
                self.timeToCZ = -1
                return
            elif present_range > target.CZ.centralRadius:
                self.timeToCZ = 0
                return
            else:
                i_run = 1
        self.compute_CPA_and_range(target)
        target_CPA_range = np.linalg.norm(target.position - self.CPA_position)  # i.e., RC in figure 7.
        if target_CPA_range > target.CZ.centralRadius:
            if target_CPA_range > target.CZ.outerCZR:
                self.timeToCZ = -1
                return
            else:
                # compute relative speed on attacker course
                relative_speed = self.compute_relative_speed(target)
                self.timeToCZ = self.CPARange / relative_speed
        else:
            angle = math.acos(target_CPA_range / target.CZ.centralRadius)
            RCZ = target.CZ.centralRadius * math.sin(angle)
            dist = -1
            match i_run:
                case 1:
                    dist = RCZ - self.CPARange
                case 2:
                    # dist = self.CPARange - RCZ
                    dist = present_range
                case 3:
                    dist = RCZ + self.CPARange
                case 0:
                    print("something wrong with i_run")
            # compute relative speed on attacker course
            assert (dist > 0)
            relative_speed = self.compute_relative_speed(target)
            self.timeToCZ = dist / relative_speed
            return

    def attacker_course_change(self):
        if self.isOnStation:        # this state can be set to False when receiving intelligence
            # reverse course
            self.course += np.pi
            if self.course > np.pi:
                self.course -= 2.0 * np.pi
            velocity_x = self.speed * math.cos(self.course)
            velocity_y = self.speed * math.sin(self.course)
            self.velocity = np.asarray([velocity_x, velocity_y])
            return self.searchLegTime
        else:
            self.isOnStation = True
            self.isInTransit = False
            self.receivedIntel = False
            self.speed = self.searchSpeed
            self.course = self.estTargetCourse - np.pi / 2.0
            if self.course < -np.pi:
                self.course += 2.0 * np.pi
            velocity_x = self.speed * math.cos(self.course)
            velocity_y = self.speed * math.sin(self.course)
            self.velocity = np.asarray([velocity_x, velocity_y])
            return self.searchLegTime / 2.0

    def target_course_change(self):
        if self.isEvading:
            self.isEvading = False
        self.legID += 1
        assert (self.legID < self.trackNUm)
        self.course = self.courseTable[self.legID]
        self.speed = self.speedTable[self.legID]
        velocity_x = self.speed * math.cos(self.course)
        velocity_y = self.speed * math.sin(self.course)
        self.velocity = np.asarray([velocity_x, velocity_y])

    def compute_detection_prob(self, target):   # refer to PDET and DETECT
        radiated_noise, self_noise = self.lookup_noise_curve()
        figure_of_merit = radiated_noise - max(self_noise, self.ambientNoiseLevel) + self.sonarGain
        prop_loss_curve = scipy.interpolate.interp1d(self.rangeCurve, self.propagationLossCurve,
                                                     fill_value='extrapolate')
        relative_track = target.position - self.position
        present_range = np.linalg.norm(relative_track)
        prop_loss = prop_loss_curve(present_range)
        z = (prop_loss - figure_of_merit) / self.STDOfPropagationLoss
        prob = 1 - scipy.stats.norm.cdf(z, loc=0, scale=self.STDOfPropagationLoss)      # a standard normal distribution
        if self.isOnStation:
            assert(not self.isInTransit)
            if not self.detectionProbDuringSearch or prob >= max(self.detectionProbDuringSearch):
                if prob > 0:
                    self.maxSearchProb = prob
                    self.occurrenceSearchMaxProb += 1
                    if self.maxSearchProb > self.maxTransitProb:
                        self.broadcastTimeMaxProb = self.timeWhenReceivingIntel
                        self.acquisitionTimeToMaxProb = self.timeOfIntel
                        self.timeOfMaxDetectionProb = self.currentTime
                        self.delaysAcquisitionToBroadcast = self.broadcastTimeMaxProb - self.acquisitionTimeToMaxProb
                        self.delaysBroadcastToMaxDetectionProb = self.timeOfMaxDetectionProb - self.broadcastTimeMaxProb
                    if target.isEvading:
                        self.occurrenceEvasionMaxProb += 1
                    self.searchProbTimes[self.currentTime] = prob
                self.detectionProbDuringSearch.append(prob)
                self.trialProbEst += (1 - self.trialProbEst) * prob
        elif self.isInTransit:
            assert(not self.isOnStation)
            if prob > 0:
                if self.transitProbTimes and prob > max(self.transitProbTimes.values()):
                    self.maxTransitProb = prob
                    self.occurrenceTransitMaxProb += 1
                    if self.maxTransitProb > self.maxSearchProb:
                        self.broadcastTimeMaxProb = self.timeWhenReceivingIntel
                        self.acquisitionTimeToMaxProb = self.timeOfIntel
                        self.timeOfMaxDetectionProb = self.currentTime
                        self.delaysAcquisitionToBroadcast = self.broadcastTimeMaxProb - self.acquisitionTimeToMaxProb
                        self.delaysBroadcastToMaxDetectionProb = self.timeOfMaxDetectionProb - self.broadcastTimeMaxProb
                    if target.isEvading:
                        self.occurrenceEvasionMaxProb += 1
                self.transitProbTimes[self.currentTime] = prob
            # if self.detectionProbDuringSearch:  # clear saved prob during last searching pattern
            #     self.detectionProbDuringSearch.clear()
            self.trialProbEst += (1 - self.trialProbEst) * prob
        else:           # This case can be that the attacker is within the convergence zone of the target
            self.trialProbEst += (1 - self.trialProbEst) * prob
        self.maxDetectionProb = max(self.maxTransitProb, self.maxSearchProb)

    def compute_time_to_interception(self):
        return np.linalg.norm(self.position - self.interceptPoint) / self.speed

    def detection_probability_event(self, target):        # for attacker's detection prob event
        if self.interceptPoint is None:
            self.compute_time_to_CPA(target)
            time_to_station = self.timeToCPA
        else:
            self.timeToInterception = self.compute_time_to_interception()
            time_to_station = self.timeToInterception
        if time_to_station <= 1e-2:
            self.isOnStation = True
            self.isInTransit = False
            self.receivedIntel = False
        else:
            self.isOnStation = False
        if self.isOnStation:
            if not self.detectionProbDuringSearch:
                return False
            if self.detectionProbDuringSearch[-1] == max(self.detectionProbDuringSearch):
                if self.trialProbEst >= 1 - 1e-2:
                    return True  # end of this trial
                else:
                    return False  # end of this event
            else:
                # drop the latest prob detection
                self.detectionProbDuringSearch.pop()
                return False
        elif self.isInTransit:
            # aggregate to form the trial detection prob.
            self.trialProbEst += (1 - self.trialProbEst) * self.detectionProbDuringSearch[-1]
            if self.trialProbEst >= 1 - 1e-2:
                return True     # end of this trial
            else:
                return False    # end of this event
        else:       # not in transit and not in searching pattern
            return False        # end of this event

    def update_evasion_course(self, relative_angle, base_course):
        if relative_angle <= np.pi:
            self.course = base_course + self.evasionAngleIncrement
            if self.course > np.pi:
                self.course -= 2.0 * np.pi
        else:
            self.course = base_course - self.evasionAngleIncrement
            if self.course <= -np.pi:
                self.course += 2.0 * np.pi

    def avoid(self, attacker):
        match self.evasionOption:
            case 1:
                self.speed /= 2.0
                self.course += math.pi
                if self.course > math.pi:
                    self.course -= 2.0 * math.pi
            case 2:
                relative_track_of_attacker = self.position - attacker.position
                bearing_of_attacker = np.arctan2(relative_track_of_attacker[1], relative_track_of_attacker[0])
                self.speed += self.evasionSpeedIncrement
                self.course = bearing_of_attacker + self.evasionAngleIncrement
                if self.course > math.pi:
                    self.course -= 2.0 * math.pi
            case 3:
                self.speed += self.evasionSpeedIncrement
                relative_track_of_target = attacker.position - self.position
                bearing_of_target = np.arctan2(relative_track_of_target[1], relative_track_of_target[0])
                relative_angle = (bearing_of_target - attacker.course) % (2 * np.pi)
                reverse_attacker_course = attacker.course + np.pi
                if reverse_attacker_course > np.pi:
                    reverse_attacker_course -= 2.0 * np.pi
                self.update_evasion_course(relative_angle, reverse_attacker_course)
            case 4:
                self.speed += self.evasionSpeedIncrement
                relative_track_of_target = attacker.position - self.position
                bearing_of_target = np.arctan2(relative_track_of_target[1], relative_track_of_target[0])
                relative_angle = (bearing_of_target - attacker.course) % (2 * np.pi)
                self.update_evasion_course(relative_angle, bearing_of_target)
            case 5:
                # do nothing, suppress counterdetection event
                self.isEvading = False
            case 6:
                # user customized evasion routine
                if self.customizedEvasionRoutine is None:
                    self.isEvading = False

    def counterdetection(self, attacker):
        if self.isEvading:
            # Cease evasion, set target course and speed
            self.isEvading = False
            self.course = self.baseCourse
            self.speed = self.baseSpeed
        else:
            self.isEvading = True
            self.baseCourse = self.course
            self.baseSpeed = self.speed
            self.avoid(attacker)

    def receive_intelligence(self, surveillance, target):
        if surveillance.newIntelReady:
            surveillance.newIntelReady = False
            self.receivedIntel = True
            self.timeWhenReceivingIntel = self.currentTime
            self.timeOfIntel = surveillance.intelTime
            self.estTargetCourse = surveillance.courseEst
            self.estTargetSpeed = surveillance.speedEst
            self.estTargetPos = surveillance.positionEst
            self.intercept(target)
            return True
        else:
            return False

    def func(self, variables):
        (theta, T) = variables
        est_target_velocity_x = self.estTargetSpeed * math.cos(self.estTargetCourse)
        est_target_velocity_y = self.estTargetSpeed * math.sin(self.estTargetCourse)
        relative_pos_x = self.estTargetPos[0] - self.position[0]
        relative_pos_y = self.estTargetPos[1] - self.position[1]
        eq1 = (est_target_velocity_y - self.speed * math.sin(theta)) * T + relative_pos_y
        eq2 = (est_target_velocity_x - self.speed * math.cos(theta)) * T + relative_pos_x
        return [eq1, eq2]

    def func_optimal(self, variables):
        (theta, T) = variables
        est_target_velocity_x = self.realTargetSpeed * math.cos(self.realTargetCourse)
        est_target_velocity_y = self.realTargetSpeed * math.sin(self.realTargetCourse)
        relative_pos_x = self.realTargetPos[0] - self.position[0]
        relative_pos_y = self.realTargetPos[1] - self.position[1]
        eq1 = (est_target_velocity_y - self.speed * math.sin(theta)) * T + relative_pos_y
        eq2 = (est_target_velocity_x - self.speed * math.cos(theta)) * T + relative_pos_x
        return [eq1, eq2]

    def intercept(self, target):        # this function can be accessed by Communication and CZ events
        self.isInTransit = True
        self.speed = self.transitSpeed
        self.realTargetCourse = target.course
        self.realTargetSpeed = target.speed
        self.realTargetPos = target.position
        result = scipy.optimize.least_squares(self.func, (0.1, 1),
                                              bounds=((-np.pi, 0), (np.pi, self.interceptTimeUpperBound)))
        optimal_result = scipy.optimize.least_squares(self.func_optimal, (0.1, 1),
                                                      bounds=((-np.pi, 0), (np.pi, self.interceptTimeUpperBound)))
        if not result.success:
            print('No interception point found.')
        else:
            self.course = result.x[0]
            self.timeToInterception = result.x[1]
            interception_pos_x = self.position[0] + self.speed * math.cos(self.course) * self.timeToInterception
            interception_pos_y = self.position[1] + self.speed * math.sin(self.course) * self.timeToInterception
            self.interceptPoint = np.asarray([interception_pos_x, interception_pos_y])
        if not optimal_result.success:
            print('No optimal interception point found.')
        else:
            optimal_course = optimal_result.x[0]
            optimal_time_to_interception = optimal_result.x[1]
            interception_pos_x = self.position[0] + self.speed * math.cos(optimal_course) * optimal_time_to_interception
            interception_pos_y = self.position[1] + self.speed * math.sin(optimal_course) * optimal_time_to_interception
            self.optimalInterceptions[self.currentTime] = np.asarray([interception_pos_x, interception_pos_y])

    def pass_convergence_zone(self, target):
        self.timeCZDetection = self.currentTime
        if self.czIntel:
            if self.receivedIntel:
                return False
            else:
                self.compute_CZ_intel(target)
                self.intercept(target)      # intercept based on the CZ intel
                return True
        else:
            self.compute_detection_prob(target)
            return False

    def compute_CZ_intel(self, target):
        np.random.seed(self.randomSeed)
        # estimate target position
        range_error = target.CZ.rangeError
        bearing_error = target.CZ.bearingError
        relative_track = target.position - self.position
        true_bearing = np.arctan2(relative_track[1], relative_track[0])
        bound1 = true_bearing - bearing_error
        if bound1 <= -np.pi:
            bound1 += 2 * np.pi
        bound2 = true_bearing + bearing_error
        if bound2 > np.pi:
            bound2 -= 2 * np.pi
        lower_bound = min(bound1, bound2)
        upper_bound = max(bound1, bound2)
        est_bearing = np.random.uniform(lower_bound, upper_bound)
        offsetPosX = range_error * math.cos(est_bearing)
        offsetPosY = range_error * math.sin(est_bearing)
        offsetPos = np.asarray([offsetPosX, offsetPosY])
        self.estTargetPos = target.position + offsetPos
        # estimate target speed
        speed_error = target.CZ.speedError
        course_error = target.CZ.courseError
        min_leg_speed = min(target.speedTable)
        assert(speed_error < min_leg_speed)     # This is for NRTT = 0. Otherwise, refer to page 97
        self.estTargetSpeed = np.random.uniform(target.speed - speed_error, target.speed + speed_error)
        # estimate target course
        bound1 = target.course - course_error
        if bound1 <= -np.pi:
            bound1 += 2 * np.pi
        bound2 = target.course + course_error
        if bound2 > np.pi:
            bound2 -= 2 * np.pi
        lower_bound = min(bound1, bound2)
        upper_bound = max(bound1, bound2)
        self.estTargetCourse = np.random.uniform(lower_bound, upper_bound)

