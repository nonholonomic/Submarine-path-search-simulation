import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow
# mpl.rcParams['animation.ffmpeg_path'] =
# r'C:\\Users\\User\\Downloads\\ffmpeg-master-latest-win64-gpl\\ffmpeg-master-latest-win64-gpl\\bin\\ffmpeg.exe'


class Drawing:
    def __init__(self):
        # fig setting
        self.env = None
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.grid()
        self.attacker_line, = self.ax.plot([], [], 'o-b', lw=2, markevery=[-1])
        self.target_line, = self.ax.plot([], [], 'v:c', lw=2, markevery=[-1])
        attacker_arrow = Arrow(0, 0, 0, 0, color='b', width=1)
        target_arrow = Arrow(0, 0, 0, 0, color='c', width=1)
        self.a_arrow = self.ax.add_patch(attacker_arrow)
        self.t_arrow = self.ax.add_patch(target_arrow)
        self.attacker_text = self.ax.text(0.02, 0.65, '', transform=self.ax.transAxes)
        self.target_text = self.ax.text(0.02, 0.60, '', transform=self.ax.transAxes)
        self.time_text = self.ax.text(0.02, 0.55, '', transform=self.ax.transAxes)
        self.isInTransit = self.ax.text(0.02, 0.40, '', transform=self.ax.transAxes)
        self.isOnStation = self.ax.text(0.02, 0.35, '', transform=self.ax.transAxes)
        self.receivedIntel = self.ax.text(0.02, 0.30, '', transform=self.ax.transAxes)
        self.isEvading = self.ax.text(0.02, 0.25, '', transform=self.ax.transAxes)
        self.interceptPoint, = self.ax.plot(0, 0, '*')
        self.interceptText = plt.annotate('interceptPoint', xy=(0, 0), xytext=(0, 0))
        # data setting
        self.attackerLineX = []
        self.attackerLineY = []
        self.targetLineX = []
        self.targetLineY = []
        self.attackerPos = None
        self.targetPos = None
        self.attackerVel = None
        self.attackerSpeed = 0
        self.targetVel = None
        self.targetSpeed = 0
        self.attackerInTransit = False
        self.attackerOnStation = False
        self.timeElapsed = 0
        self.interceptPos = None
        self.attackerReceivedIntel = False
        self.targetEvading = False

    def init(self, env, attacker, target):
        # data setting
        self.env = env
        if self.attackerLineX:
            self.attackerLineX.clear()
            self.attackerLineY.clear()
            self.targetLineX.clear()
            self.targetLineY.clear()
        self.attackerLineX.append(attacker.position[0])
        self.attackerLineY.append(attacker.position[1])
        self.attackerPos = attacker.position
        self.targetLineX.append(target.position[0])
        self.targetLineY.append(target.position[1])
        self.targetPos = target.position
        self.attackerVel = attacker.velocity
        self.attackerSpeed = attacker.speed
        self.targetVel = target.velocity
        self.targetSpeed = target.speed
        self.attackerInTransit = attacker.isInTransit
        self.attackerOnStation = attacker.isOnStation
        self.interceptPos = attacker.interceptPoint
        self.attackerReceivedIntel = attacker.receivedIntel
        self.targetEvading = target.isEvading

    def step(self, time):
        # step one by dt hours
        self.timeElapsed += time
        # update position
        self.attackerPos += time * self.attackerVel
        self.targetPos += time * self.targetVel
        self.attackerLineX.append(self.attackerPos[0])
        self.attackerLineY.append(self.attackerPos[1])
        self.targetLineX.append(self.targetPos[0])
        self.targetLineY.append(self.targetPos[1])

    def update(self, attacker, target):
        velocity_x = attacker.speed * np.math.cos(attacker.course)
        velocity_y = attacker.speed * np.math.sin(attacker.course)
        attacker.velocity = np.asarray([velocity_x, velocity_y])
        velocity_x = target.speed * np.math.cos(target.course)
        velocity_y = target.speed * np.math.sin(target.course)
        target.velocity = np.asarray([velocity_x, velocity_y])
        self.attackerVel = attacker.velocity
        self.attackerSpeed = attacker.speed
        self.targetVel = target.velocity
        self.targetSpeed = target.speed
        self.attackerInTransit = attacker.isInTransit
        self.attackerOnStation = attacker.isOnStation
        self.interceptPos = attacker.interceptPoint
        self.attackerReceivedIntel = attacker.receivedIntel
        self.targetEvading = target.isEvading

    def status_display(self):
        if self.attackerInTransit:
            self.isInTransit.set_text('Attacker Intercept')
            self.interceptPoint, = self.ax.plot(self.interceptPos[0], self.interceptPos[1], '*', color='red')
            self.interceptText = plt.annotate('interceptPoint', xy=(self.interceptPos[0], self.interceptPos[1]),
                                              xytext=(self.interceptPos[0], self.interceptPos[1]))
        else:
            self.isInTransit.set_text('')
        if self.attackerOnStation:
            self.isOnStation.set_text('Attacker Search')
        else:
            self.isOnStation.set_text('')
        if self.attackerReceivedIntel:
            self.receivedIntel.set_text('Attacker Receive Intelligence')
        else:
            self.receivedIntel.set_text('')
        if self.targetEvading:
            self.isEvading.set_text('Target Evading')
        else:
            self.isEvading.set_text('')

    def animate(self, time):
        """perform animation step"""
        self.step(time)
        # 新增对interceptPoint点清除的判断#############################
        if self.interceptPoint.figure:
            self.interceptPoint.remove()
            self.interceptText.remove()
        self.a_arrow.remove()
        self.t_arrow.remove()
        self.attacker_line.set_data(self.attackerLineX, self.attackerLineY)
        self.target_line.set_data(self.targetLineX, self.targetLineY)
        arrow_attacker = Arrow(self.attackerPos[0], self.attackerPos[1],
                               self.attackerVel[0] / self.attackerSpeed * (10 + abs(self.attackerPos[0] / 20)),
                               self.attackerVel[1] / self.attackerSpeed * (10 + abs(self.attackerPos[0] / 20)),
                               color='b', width=10)
        arrow_target = Arrow(self.targetPos[0], self.targetPos[1],
                             self.targetVel[0] / self.targetSpeed * (10 + abs(self.targetPos[0] / 20)),
                             self.targetVel[1] / self.targetSpeed * (10 + abs(self.targetPos[0] / 20)),
                             color='c', width=10)  # 对箭头坐标做一些运算，箭头显示有一定改善
        self.a_arrow = self.ax.add_patch(arrow_attacker)
        self.t_arrow = self.ax.add_patch(arrow_target)
        self.fig.canvas.draw_idle()
        self.time_text.set_text('Time = %.3f' % self.timeElapsed)
        self.attacker_text.set_text('Attacker Position: [%.3f, %.3f]' % (self.attackerPos[0], self.attackerPos[1]))
        self.target_text.set_text('Target Position: [%.3f, %.3f]' % (self.targetPos[0], self.targetPos[1]))
        self.status_display()
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
