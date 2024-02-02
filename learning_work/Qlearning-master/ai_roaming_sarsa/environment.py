# -*- coding: utf-8 -*-
class RoamEnvironment(object):

    def __init__(self):
        self.max_rssi = -62
        self.min_rssi = -75
        self.move = [[2], [1], [0], [-1], [-2]]  #
        self.nA = len(self.move)
        self.reset()

    def reset(self):
        self.th = -70
        self.score = 0
        self.done = False

    def observation(self):
        return self.th, self.done

    def clip(self, th):
        return max(min(th, self.max_rssi), self.min_rssi)

    def cal_sle(self):
        self.score = 1
        if self.th in [-75, -62]:
            self.score = 2

    def step(self, action):
        self.done = False
        self.th += self.move[action][0]
        self.th = self.clip(self.th)

        self.cal_sle()
        # note 设置不同的标准，
        #  当轻度优化的时候，+1
        #  当优化度高的时候，+2
        #  当轻度负优化时候，-1
        #  当重度负优化时候，-10（较高惩罚度）
        if self.score == 1:
            reward = -100
        elif self.score == 2:
            reward = 0
            self.is_destination = True
            self.done = True
        else:
            reward = 100

        return self.th, reward, self.done
