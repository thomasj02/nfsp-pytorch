import random


class Reservoir(object):
    def __init__(self, reservoir_size):
        self.reservoir_size = reservoir_size
        self.samples = []
        self.index = 0

    def add_sample(self, sample):
        if len(self.samples) < self.reservoir_size:
            self.samples.append(sample)
        else:
            r = random.randint(0, self.index)
            if r < self.reservoir_size:
                self.samples[r] = sample

        self.index += 1

    def sample(self, sample_size):
        return random.sample(self.samples, k=sample_size)
