import math
import bisect

import torch


class FTModel(torch.nn.Module):
    def __init__(self, models, transformation):
        super().__init__()
        self.models = {str(float(key)): value for key, value in models.items()}
        self.transformation = transformation

    def forward(self, X, aux):
        if self.transformation == 'rotate':
            aux = min(math.degrees(aux), 360.) # ensure that aux is in [0, 360]

        key_list = sorted([float(i) for i in self.models.keys()])
        pos = bisect.bisect_left(key_list, aux)

        if pos == 0:
            return key_list[0]
        if pos == len(key_list):
            return key_list[-1]

        # Compare the closest elements around the insertion point
        before = float(key_list[pos - 1])
        after = float(key_list[pos])
        if abs(before - aux) <= abs(after - aux):
            key = before
        else:
            key = after

        return self.models[str(key)](X)
