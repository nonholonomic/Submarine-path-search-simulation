import sys
import numpy as np
import math


class Surveillance:
    def __init__(self, env, setting, target):
        self.env = env
        self.randomSeed = setting.RANDOM_SEED
        self.newIntelReady = False
        self.intelID = 0      # self.intelID < self.intelNum
        # Below are lists with the size of intelNum
        if setting.INTELLIGENCE_TIME_OPTION == 0:
            self.intelTimeTable = setting.INTELLIGENCE_DETECTION_TIME
            self.intelNum = setting.INTELLIGENCE_DETECTION_NUM
        else:
            self.intelTimeTable = setting.randomlyGeneratedIntelTimes
            self.intelNum = len(setting.randomlyGeneratedIntelTimes)
        self.intelRangeErr = setting.INTELLIGENCE_DETECTION_RANGE_ERROR
        self.intelBearingEst = np.deg2rad(setting.INTELLIGENCE_BEARING_ESTIMATE)
        self.intelCourseEst = np.deg2rad(setting.INTELLIGENCE_DETECTION_COURSE_ESTIMATE)
        self.intelSpeedEst = setting.INTELLIGENCE_DETECTION_SPEED_ESTIMATE
        # End lists
        self.legNum = target.trackNUm
        self.legID = target.legID
        # Below are lists with the size of target track num
        self.posXErr = setting.SURVEILLANCE_POSITION_X_ERROR    # max error in uniform, or 2 * std in normal
        self.posYErr = setting.SURVEILLANCE_POSITION_Y_ERROR
        # End lists
        self.intelTime = 0
        self.speedErr = setting.INTELLIGENCE_SPEED_ERROR
        self.courseErr = math.radians(setting.INTELLIGENCE_COURSE_ERROR)
        self.safetyFactor = setting.SAFETY_FACTOR
        self.communicationInterval = setting.COMMUNICATION_INTERVAL
        self.fixedIntelligenceDelay = setting.FIXED_INTELLIGENCE_DELAY
        self.intelTimeOption = setting.INTELLIGENCE_TIME_OPTION
        self.intelDataOption = setting.INTELLIGENCE_DATA_OPTION
        self.intelPosErrorDistriOption = setting.INTELLIGENCE_ERROR_DISTRIBUTION_OPTION
        self.courseEst = 0
        self.speedEst = 0
        self.positionEst = np.asarray([0.0, 0.0])       # estimated target position
        self.previousPosEst = None                      # used for compute course and speed est based on positions
        self.previousPosEstTime = env.now

    def compute_intel_data(self, target):   # refer to INTEL
        # Intel time option:
        # -1 -> Random intelligence times each trial
        # 0 -> Input intelligence times
        # 1 -> Random intelligence times first trial
        # If random intelligence times
        # # Intel data option: refer to NCOMP
        # # 1 -> intel pos err distri option:   refer to INRAND
        # #      1 -> Random uniform pos est., random uniform course and speed est.
        # #      2 -> Random normal pos est., random uniform course and speed est.
        # #      3 -> Random uniform pos est., random normal course and speed est.
        # #      4 -> Random normal pos est., random normal course and speed est.
        # # 2 -> intel pos err distri option:
        # #      1, 3 -> Random uniform pos est., course and speed based on positions, use safe factor.
        # #      2, 4 -> Random normal pos est., course and speed based on positions, use safe factor.
        # Endif
        # If input intelligence times
        # # Intel data option:
        # # 3 -> intel pos err distri option:
        # #      1, 2 -> Input pos est., random uniform course and speed est.
        # #      3, 4 -> Input pos est., random normal course and speed est.
        # # 4 -> intel pos err distri option:
        # #      1--4 -> Input pos est., input course and speed est.
        # # 5 -> intel pos err distri option:
        # #      1--4 -> Input pos est., course and speed based on positions, use safe factor.
        # Endif
        np.random.seed(self.randomSeed)
        self.newIntelReady = True       # This state can be set to False when receiving intel by attacker
        currentIntelID = self.intelID
        self.intelTime = self.intelTimeTable[currentIntelID]
        self.intelID += 1
        assert(currentIntelID < self.intelNum)
        assert(abs(self.intelTime - self.env.now) < 1e-4)   # the intel time from the table should be current time
        if self.intelTimeOption == 0:   # input intel times
            # estimate target position based on range_error and bearing
            range_error = self.intelRangeErr[currentIntelID]
            bearing = self.intelBearingEst[currentIntelID]
            offsetPosX = range_error * math.cos(bearing)
            offsetPosY = range_error * math.sin(bearing)
            offsetPos = np.asarray([offsetPosX, offsetPosY])
            self.positionEst = target.position + offsetPos
            if self.intelDataOption == 3:       # Random
                self.randomly_est_course_speed(target)
            elif self.intelDataOption == 4:     # Input
                self.courseEst = self.intelCourseEst[currentIntelID]
                self.speedEst = self.intelSpeedEst[currentIntelID]
            elif self.intelDataOption == 5:     # Computed
                self.compute_course_speed_based_on_positions(target)     # refer to Fig.1
            else:
                sys.exit("Wrong Intel Data Option!")
        else:                           # random intel times
            self.randomly_est_pos(target)
            if self.intelDataOption == 1:
                self.randomly_est_course_speed(target)
            elif self.intelDataOption == 2:
                self.compute_course_speed_based_on_positions(target)
            else:
                sys.exit("Wrong Intel Data Option!")

    def compute_course_speed_based_on_positions(self, target):  # need to use the safety factor
        if self.previousPosEst is None:     # first estimate, do it randomly uniform with safety factor
            self.courseEst = target.course + np.random.uniform(-1, 1) * self.courseErr
            low_speed_bound = target.speed - self.speedErr
            high_speed_bound = target.speed + self.speedErr + self.safetyFactor
            self.speedEst = np.random.uniform(low_speed_bound, high_speed_bound)
        else:
            intervening_time = self.env.now - self.previousPosEstTime
            pos_est_track = self.positionEst - self.previousPosEst
            dist = np.linalg.norm(pos_est_track)
            self.speedEst = dist / intervening_time
            self.courseEst = np.arctan2(pos_est_track[1], pos_est_track[0])

    def randomly_est_pos(self, target):
        # generate target position estimate
        if self.intelPosErrorDistriOption == 1 or self.intelPosErrorDistriOption == 3:
            # Random uniform pos est.
            max_x_err = self.posXErr[self.legID]
            max_y_err = self.posYErr[self.legID]
            random_num = np.random.uniform(-1, 1)
            self.positionEst[0] = target.position[0] + max_x_err * random_num
            random_num = np.random.uniform(-1, 1)
            self.positionEst[1] = target.position[1] + max_y_err * random_num
        elif self.intelPosErrorDistriOption == 2 or self.intelPosErrorDistriOption == 4:
            # Random normal pos est.
            std_x_err = self.posXErr[self.legID] / 2.0
            std_y_err = self.posYErr[self.legID] / 2.0
            self.positionEst[0] = np.random.normal(target.position[0], std_x_err)
            self.positionEst[1] = np.random.normal(target.position[1], std_y_err)
        else:
            sys.exit("Wrong Intel Position Error Distribution Option!")

    def randomly_est_course_speed(self, target):
        if self.intelPosErrorDistriOption == 1 or self.intelPosErrorDistriOption == 2:
            # random uniform course and speed est.
            self.courseEst = target.course + np.random.uniform(-1, 1) * self.courseErr
            low_speed_bound = target.speed - self.speedErr
            high_speed_bound = target.speed + self.speedErr
            self.speedEst = np.random.uniform(low_speed_bound, high_speed_bound)
        elif self.intelPosErrorDistriOption == 3 or self.intelPosErrorDistriOption == 4:
            # random normal course and speed est.
            std_course_error = self.courseErr / 2.0
            std_speed_error = self.speedErr / 2.0
            self.courseEst = np.random.normal(target.course, std_course_error)
            self.speedEst = np.random.normal(target.speed, std_speed_error)
        else:
            sys.exit("Wrong Intel Position Error Distribution Option!")

    def intelligence_detection(self, target):
        # first update target leg info
        self.legID = target.legID
        assert (self.legID < self.legNum)
        self.compute_intel_data(target)
        self.previousPosEst = self.positionEst
        self.previousPosEstTime = self.env.now
        if self.communicationInterval == 0:     # 0 means continuous broadcast, send intel in a fixed interval
            return self.env.now + self.fixedIntelligenceDelay   # for communication events
        else:
            return -1
