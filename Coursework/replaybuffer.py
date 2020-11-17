from collections import deque
import random

class ReplayBuffer:
    def __init__(self, size=5000):
        self._buffer = deque(maxlen=size)
    
    def append(self, transition):
        self._buffer.append(transition)
        return self
    
    def sample(self, n):
        return random.choices(self._buffer, k=n)
    
    def __len__(self):
        return len(self._buffer)
