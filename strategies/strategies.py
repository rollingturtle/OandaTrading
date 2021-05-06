
from abc import ABCMeta, abstractmethod




class Strategy(metaclass=ABCMeta):
    @abstractmethod
    def act(self,
            position=0,
            prob_up=0.5,
            thr_up=.53,
            thr_low=.47):
        pass


class Strategy_1(Strategy):
    def __init__(self,
                 instrument,
                 order_fun,
                 report_fun,
                 live_or_test="test"
                 ):
        super().__init__()
        self.order_fun = order_fun
        self.report_fun = report_fun
        self.live_or_test = live_or_test
        self.instrument =  instrument
        self.position = 0

        return

    def act(self,
            position=0,
            prob_up=0.5,
            thr_up=.53,
            thr_low=.47,
            units=1000):

        self.position = position

        if self.live_or_test == "live":
            if self.position == 0:
                if prob_up > thr_up:  # . 0.53:
                    order = self.order_fun(self.instrument, units,suppress=True, ret=True)
                    self.report_fun(order, "GOING LONG")
                    self.position = 1
                elif prob_up < thr_low:  # 0.47:
                    order = self.order_fun(self.instrument, -units,suppress=True, ret=True)
                    self.report_fun(order, "GOING SHORT")
                    self.position = -1

            elif self.position == -1:
                if prob_up > thr_up:  # 0.53:
                    order = self.order_fun(self.instrument, units * 2,suppress=True, ret=True)
                    self.report_fun(order, "GOING LONG")
                    self.position = 1

            elif self.position == 1:
                if prob_up < thr_low:  # 0.47:
                    order = self.order_fun(self.instrument, -units * 2,suppress=True, ret=True)
                    self.report_fun(order, "GOING SHORT")
                    self.position = -1
        else:
            print("Strategy testing to be implemented!!!!")
            pass

        return self.position

