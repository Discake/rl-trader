from collections import deque

import numpy as np


class RewardFn:
    def __call__(self, prev_value, curr_value, info) -> float:
        raise NotImplementedError

class PnLReward(RewardFn):
    def __call__(self, prev_value, curr_value, info):
        return curr_value - prev_value

class LogReturnReward(RewardFn):
    def __call__(self, prev_value, curr_value, info):
        return np.log(curr_value / prev_value)

class SharpeReward(RewardFn):
    def __init__(self, window=30):
        self.returns = deque(maxlen=window)
    
    def __call__(self, prev_value, curr_value, info):
        r = (curr_value - prev_value) / prev_value
        self.returns.append(r)
        if len(self.returns) < 2:
            return r
        return np.mean(self.returns) / (np.std(self.returns) + 1e-8)

class CostAwareReward(RewardFn):
    def __call__(self, prev_value, curr_value, info):
        return (curr_value - prev_value) - info['transaction_cost']