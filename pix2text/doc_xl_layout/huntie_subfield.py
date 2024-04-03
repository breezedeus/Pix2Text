import numpy as np
import torch.utils.data as data


class Huntie_Subfield(data.Dataset):
    num_classes = 13
    num_secondary_classes = 3
    default_resolution = [768, 768]
    mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(1, 1, 3)



