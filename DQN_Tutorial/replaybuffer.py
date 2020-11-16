from collections import deque
import random

class ReplayBuffer:
    def __init__(self, size=5000):
        self._buffer = deque(maxlen=size)
    
    def add(self, transition):
        self._buffer.append(transition)
        return self
    
    def sample(self, n):
        return random.choices(self._buffer, k=n)
