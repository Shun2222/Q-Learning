import numpy as np

class Env:
    def __init__(self, size=[5, 5]):
        self.size = size
        self.map = np.zeros(size)

