import numpy as np

# Implementation from Fastgan repo
class DatasetSampler():
    def __init__(self, dataset):
        self._dataset = dataset
        
    def __iter__(self):
        dataset_length = len(self._dataset)
        index = dataset_length - 1
        order = np.random.permutation(dataset_length)
        while True:
            yield order[index]
            index += 1
            if index >= dataset_length:
                np.random.seed()
                order = np.random.permutation(dataset_length)
                index = 0
    