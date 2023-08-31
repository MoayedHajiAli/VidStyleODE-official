import numpy as np


class LambdaScheduler:
    def __init__(self, warm_up_steps, lr_min, lr_max, max_decay_steps):
        self.lr_warm_up_steps = warm_up_steps
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_max_decay_steps = max_decay_steps

    def schedule(self, n):
        if n < self.lr_warm_up_steps:
            return 0
        else:
            t = (n - self.lr_warm_up_steps) / (self.lr_max_decay_steps - self.lr_warm_up_steps)
            t = min(t, 1.0)
            lr = self.lr_min + t * (self.lr_max - self.lr_min) 
            self.last_lr = lr
            return lr

    def __call__(self, n):
        return self.schedule(n)

class ConstScheduler:
    def __init__(self, lmbd):
        self.lmbd = lmbd
    
    def __call__(self, n):
        return self.lmbd
    
