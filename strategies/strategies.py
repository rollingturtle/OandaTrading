
# collection of trading strategies
from abc import ABCMeta, abstractmethod
import numpy as np


class Strategy(metaclass=ABCMeta):
    @abstractmethod
    def act(self,
            position=0,
            prob_up=0.5,
            thr_up=.53,
            thr_low=.47,
            units=1000,
            live_or_test = "test"):
        pass


class Strategy_1(Strategy):

    def __init__(self,
                 instrument,
                 order_fun,
                 report_fun
                 ):
        super().__init__()
        self.order_fun = order_fun
        self.report_fun = report_fun
        self.instrument =  instrument
        self.position = 0
        return

    def act(self,
            position=0,
            prob_up=0.5,
            thr_up=.53,
            thr_low=.47,
            units=1000,
            predictions=None,
            live_or_test="test"):

        self.position = position

        if live_or_test == "live":
            if self.position == 0:
                if prob_up > thr_up:
                    order = self.order_fun(self.instrument, units,suppress=True, ret=True)
                    self.report_fun(order, "GOING LONG")
                    self.position = 1
                elif prob_up < thr_low:
                    order = self.order_fun(self.instrument, -units,suppress=True, ret=True)
                    self.report_fun(order, "GOING SHORT")
                    self.position = -1
            elif self.position == -1:
                if prob_up > thr_up:
                    order = self.order_fun(self.instrument, units * 2,suppress=True, ret=True)
                    self.report_fun(order, "GOING LONG")
                    self.position = 1
            elif self.position == 1:
                if prob_up < thr_low:
                    order = self.order_fun(self.instrument, -units * 2,suppress=True, ret=True)
                    self.report_fun(order, "GOING SHORT")
                    self.position = -1

            return self.position #"live"

        else: #"fw/bw test"
            positions = np.where(prob_up > thr_up, 1, np.where(prob_up < thr_low, -1,0))
            # todo: take care of 0s and make them as preceding position, do it in a better way
            for i in range(1, len(positions)):
                if positions[i] == 0:
                    positions[i] = positions[i-1]

            return positions # "test"




