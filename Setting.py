import yaml
from collections import OrderedDict
import numpy as np
import statistics


class Environment:
    def __init__(self, data_file):
        # read environment data from a yaml file
        with open(data_file, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            self.ATTACKER_SPEED = data['Attacker_speed']
            self.ATTACKER_RADIATED_NOISE = data['Attacker_radiated_noise']
            self.ATTACKER_SELF_NOISE = data['Attacker_self_noise']
            self.TARGET_SPEED = data['Target_speed']
            self.TARGET_RADIATED_NOISE = data['Target_radiated_noise']
            self.TARGET_SELF_NOISE = data['Target_self_noise']
            self.PROPAGATION_LOSS = data['Propagation_loss']
            self.RANGE_IN_MILES = data['Range_in_miles']

    def print_input_data(self):
        print('Attacker speed:', self.ATTACKER_SPEED)
        print('Attacker radiated_noise:', self.ATTACKER_RADIATED_NOISE)
        print('Attacker self noise:', self.ATTACKER_SELF_NOISE)
        print('Target speed:', self.TARGET_SPEED)
        print('Target radiated noise:', self.TARGET_RADIATED_NOISE)
        print('Target self noise:', self.TARGET_SELF_NOISE)
        print('Propagation loss:', self.PROPAGATION_LOSS)
        print('Range in miles:', self.RANGE_IN_MILES)


class Run(object):
    def __init__(self, data_file):
        self.env = None
        self.dt = 5
        self.event_calendar = OrderedDict()
        # for summary
        self.optimal_target_interception = {}
        self.counterdetections_during_transit = 0
        self.counterdetections_during_search = 0
        self.counterdetections_total = 0
        self.counterdetections_of_each_trial_transit = []
        self.counterdetections_of_each_trial_search = []
        self.counterdetections_of_each_trial_total = []
        self.times_to_end_trial = []
        self.cumulative_prob_each_trial_transit = []
        self.cumulative_prob_each_trial_search = []
        self.cumulative_prob_each_trial_total = []
        self.occurrence_max_prob_transit = 0
        self.occurrence_max_prob_search = 0
        self.occurrence_max_prob_evasion = 0
        self.max_detection_probs = []
        self.max_search_probs = []
        self.max_transit_probs = []
        self.times_of_max_detection_prob = []
        self.times_of_broadcast_max_prob = []
        self.times_of_acquisition_to_max_prob = []
        self.delays_acquisition_to_broadcast = []
        self.delays_broadcast_to_max_prob = []
        self.times_of_CZ_detection = []
        # read environment data from a yaml file
        with open(data_file, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            self.TRIAL_NUM = data['Trial_num']
            self.TARGET_TRACK_NUM = data['Target_track_num']
            self.INTELLIGENCE_DETECTION_NUM = data['Intelligence_detection_num']
            self.OUTPUT_LEVEL = data['Output_level']
            self.INTELLIGENCE_TIME_OPTION = data['Intelligence_time_option']
            self.RANDOM_SEED = data['Random_seed']
            self.EVASION_OPTION = data['Evasion_option']
            self.INTELLIGENCE_DATA_OPTION = data['Intelligence_data_option']
            self.INTELLIGENCE_ERROR_DISTRIBUTION_OPTION = data['Intelligence_error_distribution_option']
            self.NUMBER_OF_CHANGE_TO_SITUATION = data['Number_of_changes_to_situation']
            self.RUN_SUMMARY_SUPPRESSION = data['Run_summary_suppression']
            self.RANDOM_TARGET_TRACK = data['Random_target_track']
            self.CONVERGENCE_ZONE_INTELLIGENCE = data['Convergence_zone_intelligence']
            self.NOMINAL_TIME_TO_END_THE_TRIAL = data['Nominal_time_to_end_the_trial']
            self.COMMUNICATION_INTERVAL = data['Communication_interval']
            self.X_TARGET_INIT = data['X_target_init']
            self.Y_TARGET_INIT = data['Y_target_init']
            self.FIXED_INTELLIGENCE_DELAY = data['Fixed_intelligence_delay']
            self.INTELLIGENCE_DETECTION_INTERVAL = data['Intelligence_detection_interval']
            self.EVASION_TIME = data['Evasion_time']
            self.SEARCH_SPEED = data['Search_speed']
            self.TRANSIT_SPEED = data['Transit_speed']
            self.ATTACKER_INITIAL_SPEED = data['Attacker_initial_speed']
            self.ATTACKER_INITIAL_COURSE = data['Attacker_initial_course']
            self.SEARCH_LEG_TIME = data['Search_leg_time']
            self.AMBIENT_NOISE_LEVEL = data['Ambient_noise_level']
            self.STD_OF_PROPAGATION_LOSS = data['STD_of_propagation_loss']
            self.TARGET_EVASION_ANGLE = data['Target_evasion_angle']
            self.EVASION_SPEED_INCREMENT = data['Evasion_speed_increment']
            self.INTELLIGENCE_SPEED_ERROR = data['Intelligence_speed_error']
            self.INTELLIGENCE_COURSE_ERROR = data['Intelligence_course_error']
            self.SAFETY_FACTOR = data['Safety_factor']
            self.ATTACKER_SONAR_GAIN = data['Attacker_sonar_gain']
            self.TARGET_SONAR_GAIN = data['Target_sonar_gain']
            self.CONVERGENCE_ZONE_CENTRAL_RADIUS = data['Convergence_zone_central_radius']
            self.CONVERGENCE_ZONE_HALF_WIDTH = data['Convergence_zone_half_width']
            self.CONVERGENCE_ZONE_SPEED_ERROR = data['Convergence_zone_speed_error']
            self.CONVERGENCE_ZONE_COURSE_ERROR = data['Convergence_zone_course_error']
            self.CONVERGENCE_ZONE_BEARING_ERROR = data['Convergence_zone_bearing_error']
            self.CONVERGENCE_ZONE_RANGE_ERROR = data['Convergence_zone_range_error']
            self.TARGET_BASE_COURSE = data['Target_base_course']
            self.TARGET_BASE_SPEED = data['Target_base_speed']
            self.TERMINAL_TIME = data['Terminal_time']
            self.INTELLIGENCE_DETECTION_PROB = data['Intelligence_detection_prob']
            self.SURVEILLANCE_POSITION_X_ERROR = data['Surveillance_position_x_error']
            self.SURVEILLANCE_POSITION_Y_ERROR = data['Surveillance_position_y_error']
            self.INTELLIGENCE_DETECTION_TIME = data['Intelligence_detection_time']
            self.INTELLIGENCE_DETECTION_RANGE_ERROR = data['Intelligence_detection_range_error']
            self.INTELLIGENCE_BEARING_ESTIMATE = data['Intelligence_bearing_estimate']
            self.INTELLIGENCE_DETECTION_COURSE_ESTIMATE = data['Intelligence_detection_course_estimate']
            self.INTELLIGENCE_DETECTION_SPEED_ESTIMATE = data['Intelligence_detection_speed_estimate']
            self.randomlyGeneratedIntelTimes = []
            self.trialProbEstimates = []    # its size is the Trial_num
            self.initial_range = 500
            self.range = -1

    def generate_random_intelligence_times(self):
        # this is based on TINT and PINT
        count_time = 0
        detection_index = 0
        while count_time < self.NOMINAL_TIME_TO_END_THE_TRIAL:
            for i in range(self.TARGET_TRACK_NUM):
                if count_time <= self.TERMINAL_TIME[i]:
                    detect_prob = self.INTELLIGENCE_DETECTION_PROB[i]
                    random_num = np.random.uniform(0, 1)
                    if random_num <= detect_prob:
                        self.randomlyGeneratedIntelTimes[detection_index] = count_time
                        detection_index += 1
                    count_time += self.INTELLIGENCE_DETECTION_INTERVAL
                else:
                    continue

    def event_insertion(self, key, event_type):
        if key in self.event_calendar.keys():
            key += 1e-6
            self.event_calendar[key] = event_type
        else:
            self.event_calendar[key] = event_type

    def establish_event_calendar(self, attacker, target, events):
        # 1st event: Target_Course_Change
        for i in range(self.TARGET_TRACK_NUM):
            self.event_insertion(self.TERMINAL_TIME[i], events[0])

        # 2nd event: Counterdetection
        if target.counterdetectionTime != -1:
            self.event_insertion(target.counterdetectionTime, events[1])

        # 3rd event: Intelligent_Detection
        # this event depends on the intel time option, refer to NOPT = -1, 0, 1
        if self.INTELLIGENCE_TIME_OPTION == 0:  # from input data
            for i in range(self.INTELLIGENCE_DETECTION_NUM):
                self.event_insertion(self.INTELLIGENCE_DETECTION_TIME[i], events[2])
        else:                                   # from randomly generated data
            for i in range(len(self.randomlyGeneratedIntelTimes)):
                self.event_insertion(self.randomlyGeneratedIntelTimes[i], events[2])

        # 4th event: Convergence_Zone
        if attacker.timeToCZ != -1:
            self.event_insertion(attacker.timeToCZ, events[3])

        # 5th event: Communications
        if self.COMMUNICATION_INTERVAL == 0:
            for i in range(self.INTELLIGENCE_DETECTION_NUM):
                count_time = self.INTELLIGENCE_DETECTION_TIME[i] + self.FIXED_INTELLIGENCE_DELAY
                while count_time < self.NOMINAL_TIME_TO_END_THE_TRIAL:
                    self.event_insertion(count_time, events[4])
                    count_time += self.FIXED_INTELLIGENCE_DELAY
        else:
            count_time = self.COMMUNICATION_INTERVAL
            while count_time < self.NOMINAL_TIME_TO_END_THE_TRIAL:
                self.event_insertion(count_time, events[4])
                count_time += self.COMMUNICATION_INTERVAL

        # 6th event: Detection_Probability
        if attacker.timeToCPA != -1:
            self.event_insertion(attacker.timeToCPA, events[5])

            # 7th event: Attacker_Course_Change
            self.event_insertion(attacker.timeToCPA, events[6])
            count_time = attacker.timeToCPA
            first_search = True
            intel_interrupt = False
            for i in range(self.INTELLIGENCE_DETECTION_NUM):
                intel_detection_time = self.INTELLIGENCE_DETECTION_TIME[i]
                while count_time < intel_detection_time:
                    intel_interrupt = True
                    if first_search:
                        first_search = False
                        count_time += 0.5 * self.SEARCH_LEG_TIME
                        if count_time < intel_detection_time:
                            self.event_insertion(count_time, events[6])
                        else:
                            break
                        count_time += self.SEARCH_LEG_TIME
                    else:
                        self.event_insertion(count_time, events[6])
                        count_time += self.SEARCH_LEG_TIME
                if intel_interrupt:
                    break

        # 8th event: End_of_Trial
        self.event_insertion(self.NOMINAL_TIME_TO_END_THE_TRIAL - 0.1, events[7])
        self.event_calendar = OrderedDict(sorted(self.event_calendar.items()))

    def initialization(self, env, attacker, target, surveillance, trial_id, events, drawing):
        self.env = env
        self.initial_range = np.linalg.norm(attacker.position - target.position)
        attacker.compute_time_to_CPA(target)
        # counterdetection occurs for the target while detection occurs for the attacker
        target.compute_counterdetection_time(attacker)
        attacker.compute_time_to_CZ(target)
        self.establish_event_calendar(attacker, target, events)
        env.process(self.start_trial(attacker, target, surveillance, trial_id, drawing))

    def update_event_calendar(self, current_time, new_event_time, event_type):
        # find the most recent event of the event_type
        # first check if the new event time and the event type is already in the calendar
        if new_event_time in self.event_calendar:
            if self.event_calendar[new_event_time] == event_type:
                return
            else:
                new_event_time += 1e-3
        # then, delete all events of the event type prior to new_event_time
        for key in list(self.event_calendar):
            assert (key >= current_time)
            if self.event_calendar[key] == event_type:
                del self.event_calendar[key]
                if key > new_event_time:
                    break
        # then, insert the new event time to the dict
        self.event_calendar[new_event_time] = event_type
        self.event_calendar = OrderedDict(sorted(self.event_calendar.items()))

    def revise_event_times(self, attacker, target, surveillance, current_time, event_type):
        if target.approachingAttacker:
            attacker.compute_time_to_CPA(target)
            if attacker.isInTransit:
                new_detection_event_time = attacker.currentTime + attacker.timeToInterception
            else:
                new_detection_event_time = attacker.currentTime + attacker.timeToCPA
            self.update_event_calendar(current_time, new_detection_event_time, 'Detection_Probability')
            self.check_on_station(current_time, attacker, target)
        else:
            if target.wasApproachingAttacker:
                new_detection_event_time = self.env.now
                self.update_event_calendar(current_time, new_detection_event_time, 'Detection_Probability')
            else:
                self.check_on_station(current_time, attacker, target)
        self.end_of_event(event_type, attacker, target, surveillance)

    def check_on_station(self, current_time, attacker, target):
        if attacker.isOnStation:
            attacker.compute_detection_prob(target)
        if self.CONVERGENCE_ZONE_INTELLIGENCE:
            attacker.compute_time_to_CZ(target)
            if attacker.timeToCZ != -1:
                self.update_event_calendar(current_time, attacker.timeToCZ + current_time, 'Convergence_Zone')
        # self.EVASION_OPTION == 5 means that there is no evasion, counterdetection is suppressed.
        if self.EVASION_OPTION != 5 and not target.isEvading:
            target.compute_counterdetection_time(attacker)
            if target.counterdetectionTime >= 0:
                self.update_event_calendar(current_time, target.counterdetectionTime + current_time, 'Counterdetection')

    def print_submarines_state(self, attacker, target):
        print('Attacker course:', attacker.course)
        print('Attacker speed:', attacker.speed)
        print('Target course:', target.course)
        print('Target speed:', target.speed)
        print('Target coordinates:', target.position)
        print('Attacker coordinates:', attacker.position)
        print('Range:', self.range)

    def end_of_event(self, event_type, attacker, target, surveillance):
        # print event data according to the event type
        self.range = np.linalg.norm(attacker.position - target.position)
        if attacker.detectionProbDuringSearch:
            current_prob = attacker.detectionProbDuringSearch[-1]
        else:
            current_prob = 0
        match event_type:
            case 'Target_Course_Change':
                print('Problem time:', self.env.now)
                print('New Target Leg')
                print('Current intelligence time:', surveillance.intelTime)
                print('Expected CPA time:', attacker.timeToCPA)
                print('Current probability of detection:', current_prob)
                print('Cumulative probability of detection:', attacker.trialProbEst)
                if not attacker.isOnStation:
                    self.print_submarines_state(attacker, target)
            case 'Intelligent_Detection':
                print('Problem time:', self.env.now)
                print('Intelligent Detection')
                print('Intelligence position:', surveillance.positionEst)
                print('Intelligence course:', surveillance.courseEst)
                print('Intelligence speed:', surveillance.speedEst)
                print('Current intelligence time:', surveillance.intelTime)
                print('Expected CPA time:', attacker.timeToCPA)
                print('Cumulative probability of detection:', attacker.trialProbEst)
                if not attacker.isOnStation:
                    self.print_submarines_state(attacker, target)
            case 'Convergence_Zone':
                print('Problem time:', self.env.now)
                print('Convergence Zone')
                print('Current intelligence time:', surveillance.intelTime)
                print('Expected CPA time:', attacker.timeToCPA)
                print('Current probability of detection:', current_prob)
                print('Cumulative probability of detection:', attacker.trialProbEst)
                if not attacker.isOnStation:
                    self.print_submarines_state(attacker, target)
            case 'Communications':
                print('Problem time:', self.env.now)
                print('Communications')
                print('Current intelligence time:', surveillance.intelTime)
                print('Expected CPA time: ', attacker.timeToCPA)
                print('Current probability of detection:', current_prob)
                print('Cumulative probability of detection:', attacker.trialProbEst)
                print('Updated intelligence coordinates: ')
                if not attacker.isOnStation:
                    self.print_submarines_state(attacker, target)
                pass
            # case 'Detection_Probability':
            case 'Counterdetection':
                print('Problem time:', self.env.now)
                print('Counterdetection')
                print('Current intelligence time:', surveillance.intelTime)
                if not target.approachingAttacker:
                    print('Current probability of detection:', current_prob)
                    print('Cumulative probability of detection:', attacker.trialProbEst)
                if attacker.isOnStation:
                    self.print_submarines_state(attacker, target)
            case 'Attacker_Course_Change':
                print('Problem time: ', self.env.now)
                print('Attacker Course Change')
                print('Current intelligence time:', surveillance.intelTime)
                if not target.approachingAttacker:
                    print('Current probability of detection:', current_prob)
                    print('Cumulative probability of detection:', attacker.trialProbEst)
                if attacker.isOnStation:
                    self.print_submarines_state(attacker, target)
            case 'End_of_Trial':
                print('Problem time: ', self.env.now)
                print('Current intelligence time: ', surveillance.intelTime)
                if not target.approachingAttacker:
                    print('Range opening.')
                    print('Current probability of detection:', current_prob)
                    print('Cumulative probability of detection:', attacker.trialProbEst)
                if attacker.isOnStation:
                    print('Attacker on station.')
                    self.print_submarines_state(attacker, target)

    def print_trial_summary(self, trial_id, attacker):
        print('End of trial summary of detection probabilities -- Trial ', trial_id)
        print('Cumulative Probability of detection:', attacker.trialProbEst)
        if attacker.transitProbTimes:
            print('Transit Probability -> Time: Probability', attacker.transitProbTimes)
        if attacker.searchProbTimes:
            print('On State Probability -> Time: Probability', attacker.searchProbTimes)
        if not attacker.transitProbTimes and not attacker.searchProbTimes:
            print('No positive detection probability established.')
        print('Random seed: ', self.RANDOM_SEED)

    def print_run_summary(self):
        print('End of Game.')
        print('Total of', self.TRIAL_NUM, 'Trials.')
        print('Initial Range to Target:', self.initial_range, 'miles.')
        if not self.optimal_target_interception:
            print('Impossible to be intercepted in any time given the setting.')
        else:
            print('Optimal target intercept under perfect information -> Time: Position')
            print(self.optimal_target_interception)
        print(' ')
        print('Counterdetections of attacker by target each trial')
        print('During transit:')
        print(self.counterdetections_of_each_trial_transit)
        print('Sample mean:', statistics.mean(self.counterdetections_of_each_trial_transit),
              '; Sample variance:', statistics.variance(self.counterdetections_of_each_trial_transit))
        print(' ')
        print('During search:')
        print(self.counterdetections_of_each_trial_search)
        print('Sample mean:', statistics.mean(self.counterdetections_of_each_trial_search),
              '; Sample variance:', statistics.variance(self.counterdetections_of_each_trial_search))
        print(' ')
        print('Total:')
        print(self.counterdetections_of_each_trial_total)
        print('Sample mean:', statistics.mean(self.counterdetections_of_each_trial_total),
              '; Sample variance:', statistics.variance(self.counterdetections_of_each_trial_total))
        print(' ')
        print('Times to end trial:')
        print(self.times_to_end_trial)
        print('Sample mean:', statistics.mean(self.times_to_end_trial),
              '; Sample variance:', statistics.variance(self.times_to_end_trial))
        print(' ')
        print('Cumulative probabilities each trial:')
        print(self.cumulative_prob_each_trial_total)
        print('Sample mean:', statistics.mean(self.cumulative_prob_each_trial_total),
              '; Sample variance:', statistics.variance(self.cumulative_prob_each_trial_total))
        print(' ')
        print('Cumulative transit probabilities each trial:')
        print(self.cumulative_prob_each_trial_transit)
        print('Sample mean:', statistics.mean(self.cumulative_prob_each_trial_transit),
              '; Sample variance:', statistics.variance(self.cumulative_prob_each_trial_transit))
        print(' ')
        print('Cumulative station probabilities each trial:')
        print(self.cumulative_prob_each_trial_search)
        print('Sample mean:', statistics.mean(self.cumulative_prob_each_trial_search),
              '; Sample variance:', statistics.variance(self.cumulative_prob_each_trial_search))
        print(' ')
        print('Number of occurrences of maximum detection probability during:')
        print('Transit:', self.occurrence_max_prob_transit)
        print('Search:', self.occurrence_max_prob_search)
        print('Evasion:', self.occurrence_max_prob_evasion)
        print(' ')
        print('Maximum Detection Probabilities')
        print(self.max_detection_probs)
        print('Sample mean:', statistics.mean(self.max_detection_probs),
              '; Sample variance:', statistics.variance(self.max_detection_probs))
        print(' ')
        print('Maximum Search Probabilities')
        print(self.max_search_probs)
        print('Sample mean:', statistics.mean(self.max_search_probs),
              '; Sample variance:', statistics.variance(self.max_search_probs))
        print(' ')
        print('Maximum Transit Probabilities')
        print(self.max_transit_probs)
        print('Sample mean:', statistics.mean(self.max_transit_probs),
              '; Sample variance:', statistics.variance(self.max_transit_probs))
        print(' ')
        print('Times of Maximum Detection Probabilities')
        print(self.times_of_max_detection_prob)
        print('Sample mean:', statistics.mean(self.times_of_max_detection_prob),
              '; Sample variance:', statistics.variance(self.times_of_max_detection_prob))
        num_positives = sum(1 for i in self.times_of_max_detection_prob if i >= 0)
        print('Total positive entries:', num_positives)
        print(' ')
        print('Times of Last Broadcast Before Maximum Probabilities')
        print(self.times_of_broadcast_max_prob)
        print('Sample mean:', statistics.mean(self.times_of_broadcast_max_prob),
              '; Sample variance: ', statistics.variance(self.times_of_broadcast_max_prob))
        num_positives = sum(1 for i in self.times_of_broadcast_max_prob if i >= 0)
        print('Total positive entries:', num_positives)
        print(' ')
        print('Acquisition Times of Intelligence Leading to Maximum Probabilities')
        print(self.times_of_acquisition_to_max_prob)
        print('Sample mean:', statistics.mean(self.times_of_acquisition_to_max_prob),
              '; Sample variance:', statistics.variance(self.times_of_acquisition_to_max_prob))
        num_positives = sum(1 for i in self.times_of_acquisition_to_max_prob if i >= 0)
        print('Total positive entries:', num_positives)
        print(' ')
        print('Delays from Acquisition to Broadcast')
        print(self.delays_acquisition_to_broadcast)
        print('Sample mean: ', statistics.mean(self.delays_acquisition_to_broadcast),
              '; Sample variance: ', statistics.variance(self.delays_acquisition_to_broadcast))
        num_positives = sum(1 for i in self.delays_acquisition_to_broadcast if i >= 0)
        print('Total positive entries:', num_positives)
        print(' ')
        print('Delays from Broadcast to Maximum Probabilities')
        print(self.delays_broadcast_to_max_prob)
        print('Sample mean:', statistics.mean(self.delays_broadcast_to_max_prob),
              '; Sample variance:', statistics.variance(self.delays_broadcast_to_max_prob))
        num_positives = sum(1 for i in self.delays_broadcast_to_max_prob if i >= 0)
        print('Total positive entries:', num_positives)
        print(' ')
        print('Convergence Zone Detection Times')
        print(self.times_of_CZ_detection)

    def compute_cumulative_prob_each_trial(self, attacker):
        cum_prob_transit = 0
        for key, value in attacker.transitProbTimes.items():
            cum_prob_transit += (1 - cum_prob_transit) * value
        self.cumulative_prob_each_trial_transit.append(cum_prob_transit)
        cum_prob_search = 0
        for key, value in attacker.searchProbTimes.items():
            cum_prob_search += (1 - cum_prob_search) * value
        self.cumulative_prob_each_trial_search.append(cum_prob_search)
        cum_prob_total = cum_prob_transit + cum_prob_search
        self.cumulative_prob_each_trial_total.append(cum_prob_total)

    def end_of_trial(self, attacker, target, trial_id):
        attacker.compute_detection_prob(target)
        self.times_of_CZ_detection.append(attacker.timeCZDetection)
        self.times_of_max_detection_prob.append(attacker.timeOfMaxDetectionProb)
        self.times_of_broadcast_max_prob.append(attacker.broadcastTimeMaxProb)
        self.times_of_acquisition_to_max_prob.append(attacker.acquisitionTimeToMaxProb)
        self.delays_broadcast_to_max_prob.append(attacker.delaysBroadcastToMaxDetectionProb)
        self.delays_acquisition_to_broadcast.append(attacker.delaysAcquisitionToBroadcast)
        self.occurrence_max_prob_evasion = self.occurrence_max_prob_evasion + attacker.occurrenceEvasionMaxProb
        self.occurrence_max_prob_search = self.occurrence_max_prob_search + attacker.occurrenceSearchMaxProb
        self.occurrence_max_prob_transit = self.occurrence_max_prob_transit + attacker.occurrenceTransitMaxProb
        self.max_search_probs.append(attacker.maxSearchProb)
        self.max_transit_probs.append(attacker.maxTransitProb)
        self.max_detection_probs.append(attacker.maxDetectionProb)
        self.times_to_end_trial.append(self.env.now)
        self.compute_cumulative_prob_each_trial(attacker)
        self.counterdetections_total = self.counterdetections_during_search + self.counterdetections_during_transit
        self.counterdetections_of_each_trial_search.append(self.counterdetections_during_search)
        self.counterdetections_of_each_trial_transit.append(self.counterdetections_during_transit)
        self.counterdetections_of_each_trial_total.append(self.counterdetections_total)
        for key, value in attacker.optimalInterceptions.items():
            self.optimal_target_interception[key] = value
        if trial_id != self.TRIAL_NUM-1:
            # print trial summary, refer to page 70, figure 15
            self.print_trial_summary(trial_id, attacker)
        else:
            # print run summary, refer to page 71, figure 16
            self.print_run_summary()
        # clear run summary counters for each trial
        self.counterdetections_during_transit = 0
        self.counterdetections_during_search = 0
        self.counterdetections_total = 0

    def start_trial(self, attacker, target, surveillance, trial_id, drawing):
        self.range = np.linalg.norm(attacker.position - target.position)
        print('Start trial at', self.env.now)
        print('Attacker course:', attacker.course)
        print('Attacker speed:', attacker.speed)
        print('Target course:', target.course)
        print('Target speed:', target.speed)
        print('Target coordinate:', target.position)
        print('Range between units:', self.range)
        while True:
            if not self.event_calendar:
                continue
            first_item = self.event_calendar.popitem(False)
            key = first_item[0]
            value = first_item[1]
            is_drawing = True
            while is_drawing:
                if key - self.env.now >= self.dt:
                    yield self.env.timeout(self.dt)
                    drawing.animate(self.dt)
                    attacker.currentTime = self.env.now
                    target.currentTime = self.env.now
                    continue
                else:
                    is_drawing = False
                    event_duration = key - self.env.now
                    yield self.env.timeout(event_duration)
                    drawing.animate(event_duration)
                    attacker.currentTime = self.env.now
                    target.currentTime = self.env.now
                    attacker.compute_time_to_CPA(target)
                    target.compute_counterdetection_time(attacker)
                    attacker.compute_time_to_CZ(target)
                    if self.CONVERGENCE_ZONE_INTELLIGENCE:
                        if attacker.timeToCZ != -1 and not ('Convergence_Zone' in self.event_calendar.values()):
                            self.update_event_calendar(self.env.now, attacker.timeToCZ + self.env.now,
                                                       'Convergence_Zone')
                    # self.EVASION_OPTION == 5 means that there is no evasion, counterdetection is suppressed.
                    if self.EVASION_OPTION != 5 and not target.isEvading:
                        if target.counterdetectionTime >= 0:
                            self.update_event_calendar(self.env.now, target.counterdetectionTime + self.env.now,
                                                       'Counterdetection')
                    attacker.compute_detection_prob(target)
                match value:
                    case 'Target_Course_Change':
                        print('Target course change at', self.env.now)
                        target.target_course_change()
                        self.revise_event_times(attacker, target, surveillance, key, value)
                    case 'Counterdetection':
                        print('Target counterdetection at', self.env.now)
                        target.counterdetection(attacker)
                        if attacker.isInTransit:
                            self.counterdetections_during_transit += 1
                        elif attacker.isOnStation:
                            self.counterdetections_during_search += 1
                        if target.isEvading:    # reschedule counterdetection event with evasion interval
                            next_counterdetection_time = self.env.now + target.evasionTime
                            self.update_event_calendar(key, next_counterdetection_time, 'Counterdetection')
                        else:
                            if target.counterdetectionTime != -1:
                                next_counterdetection_time = self.env.now + target.counterdetectionTime
                                self.update_event_calendar(key, next_counterdetection_time, 'Counterdetection')
                    case 'Intelligent_Detection':
                        print('Surveillance intelligent detection at', self.env.now)
                        next_communication_time = surveillance.intelligence_detection(target)
                        if next_communication_time != -1:   # -1 means that not continuous broadcast
                            self.update_event_calendar(key, next_communication_time, 'Communications')
                        self.end_of_event(value, attacker, target, surveillance)
                    case 'Convergence_Zone':
                        print('Attacker passes convergence zone at', self.env.now)
                        CZ_intel = attacker.pass_convergence_zone(target)
                        if not CZ_intel:
                            self.end_of_event(value, attacker, target, surveillance)
                        else:
                            self.revise_event_times(attacker, target, surveillance, key, value)
                    case 'Communications':
                        print('Attacker receives the intelligence from the surveillance at', self.env.now)
                        new_intel_ready = attacker.receive_intelligence(surveillance, target)
                        if not new_intel_ready:
                            self.end_of_event(value, attacker, target, surveillance)
                        else:
                            # update attacker course change event
                            next_event_time = self.env.now + attacker.timeToInterception
                            self.update_event_calendar(key, next_event_time, 'Attacker_Course_Change')
                            self.revise_event_times(attacker, target, surveillance, key, value)
                    case 'Detection_Probability':
                        print('Attacker compute its detection probability at', self.env.now)
                        is_end_of_trial = attacker.detection_probability_event(target)
                        if is_end_of_trial:
                            self.end_of_trial(attacker, target, trial_id)
                        else:
                            self.end_of_event(value, attacker, target, surveillance)
                    case 'Attacker_Course_Change':
                        print('Attacker course change at', self.env.now)
                        next_search_leg_time = attacker.attacker_course_change()
                        next_event_time = self.env.now + next_search_leg_time
                        self.update_event_calendar(key, next_event_time, 'Attacker_Course_Change')
                    case 'End_of_Trial':
                        print('End of trial %d at %f' % (trial_id, self.env.now))
                        self.end_of_event(value, attacker, target, surveillance)
                        self.end_of_trial(attacker, target, trial_id)
                        # early terminal the trial
                        rest_time = self.NOMINAL_TIME_TO_END_THE_TRIAL - self.env.now
                        yield self.env.timeout(rest_time)
                drawing.update(attacker, target)
