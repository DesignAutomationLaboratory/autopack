import numpy as np

class CostFieldTemplate:
    def __init__(self, size=[0,0,0]):
        self.size = size
        self.coordinates = np.zeros((size[0], size[1], size[2], 3))
    def set_coords_from_str_array(self, coord_array):
        n=0
        for i in range(self.size[0]):
            for ii in range(self.size[1]):
                for iii in range(self.size[2]):
                    self.coordinates[i,ii,iii] = [float(coord_array[n]), float(coord_array[n+1]), float(coord_array[n+2])]
                    n = n + 3

class CostField:
    def __init__(self, template):
        self.template = template
        self.costs = np.zeros((self.template.size[0], self.template.size[1], self.template.size[2], 1))
    def set_costs_from_str_array(self, cost_array):
        n=0
        for i in range(self.template.size[0]):
            for ii in range(self.template.size[1]):
                for iii in range(self.template.size[2]):
                    try:
                        if float(cost_array[n]) > 9999999:
                            cost = 999999999999999999999
                        else:
                            cost = float(cost_array[n])
                    except ValueError:
                        cost = 999999999999999999999
                    self.costs[i,ii,iii] = cost
                    n = n + 1
    def get_cost_field_as_str(self):
        cost_str = ""
        for i in range(self.template.size[0]):
            for ii in range(self.template.size[1]):
                for iii in range(self.template.size[2]):
                    cost_str = cost_str + "," + str(i) + "," + str(ii) + "," + str(iii) + "," + str(self.costs[i,ii,iii][0])
        return cost_str[1:]

