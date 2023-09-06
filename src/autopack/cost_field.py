import numpy as np

class CostFieldTemplate:
    def __init__(self, size=[0,0,0]):
        self.size = size
        self.coordinates = np.zeros((size[0], size[1], size[2], 3))

class CostField:
    def __init__(self, template):
        self.template = template
        self.costs = np.zeros((template.coordinates.size[0], template.coordinates.size[1], template.coordinates.size[2], 1))

