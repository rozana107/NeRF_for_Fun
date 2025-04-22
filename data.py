""" Dataset Classes """ 

import os
import re
import torch
from torch.utils.data import Dataset
from PIL import Image


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, pattern = 'r"(.*)_(.*).png"'):
        self.root_dir = root_dir
        self.pattern = re.compile(pattern)
        self.files = []
        for file in os.listdir(self.root_dir):
            self.files.append((file, self.root_dir))


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        file_path = os.path.join(self.root_dir, file)
        image = Image.open(file_path)
        # Apply any preprocessing or transformations here
        image = image.convert('RGB')
        match = self.pattern.search(file)
        theta = float(match.group(1)) 
        phi = float(match.group(2)) 
        return image, theta, phi

