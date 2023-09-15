import simpy
import Submarine
import Surveillance
from Setting import Environment, Run
import Drawing
import matplotlib.pyplot as plt

# Simulation configuration
NUM_RUNS = 1
events = [
    'Target_Course_Change',      # occurs at the next leg of the target
    'Counterdetection',          # occurs when the target detects counterdetection
    'Intelligent_Detection',     # occurs when the surveillance facility detects the target
    'Convergence_Zone',          # occurs when the attacker passes the CZ of the target
    'Communications',            # occurs periodically referring to Communication_interval.
                                 # If Communication_interval = 0, referring to Fixed_intelligence_delay
    'Detection_Probability',     # occurs when reaching the intercept point, when course change,
                                 # when achieve max prob during that search
    'Attacker_Course_Change',    # occurs upon arrival at search station (intercept) and at the end of each search leg
    'End_of_Trial'               # occurs when no further positive detection probabilities or
                                 # when trial time exceeds a nominal termination time
]
env_setting = Environment('env_data.yaml')
run_setting = Run('run_data_0.yaml')
# init attacker, target, and surveillance facility
drawing = Drawing.Drawing()
# running script
for run_id in range(NUM_RUNS):
    for trial_id in range(run_setting.TRIAL_NUM):
        env = simpy.Environment()
        trial_end_time = run_setting.NOMINAL_TIME_TO_END_THE_TRIAL
        attacker = Submarine.Submarine(env, run_setting, 1, env_setting)
        target = Submarine.Submarine(env, run_setting, 0, env_setting)
        surveillance = Surveillance.Surveillance(env, run_setting, target)
        drawing.init(env, attacker, target)
        if run_setting.INTELLIGENCE_TIME_OPTION == -1:     # randomly generate for each trial
            run_setting.generate_random_intelligence_times()
            if trial_id == 0:
                env_setting.print_input_data()
        elif run_setting.INTELLIGENCE_TIME_OPTION == 1:    # only randomly generate for the first trial
            if trial_id == 0:
                run_setting.generate_random_intelligence_times()
                env_setting.print_input_data()
        else:
            env_setting.print_input_data()
        run_setting.initialization(env, attacker, target, surveillance, trial_id, events, drawing)
        print('Trial %d' % trial_id)
        env.run(until=trial_end_time)
        plt.show()
