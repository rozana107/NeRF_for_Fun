""" Dataset Classes """ 

import os
import csv
from torch.utils.data import Dataset
from PIL import Image


class SyntheticDataset(Dataset):
    def __init__(self, 
        root: str, 
        csv_filepath: str):

        self.root = root
        self.img_width = 128
        self.img_height = 128

        self.snapshots = []
        # Open the input CSV file for reading
        with open(csv_filepath, 'r') as csvfile:
            reader = csv.reader(csvfile)
            # Read each row and extract the values
            for row in reader:
                self.snapshots += row

    def __len__(self):
        return len(self.snapshots)* self.img_width*self.img_height

    def __getitem__(self, idx):

        # Calculate the snapshot index
        snapshot_idx = idx // (self.img_width * self.img_height)
        # calculate the corresponding snapshot
        snapshot = self.snapshots[snapshot_idx]

        # Local index within the snapshot
        pixel_index = idx % (self.img_width * self.img_height)
        pixel_x = pixel_index % self.img_width
        pixel_y = pixel_index // self.img_width

        # Extract the values from the row
        image_filename = snapshot[0]
        camera_position = [float(snapshot[i]) for i in range(1, 4)]
        camera_orientation = [float(snapshot[i]) for i in range(4, 13)]
        theta = float(snapshot[13])
        phi = float(snapshot[14])


        # open the image
        image = Image.open(os.path.join(self.root, image_filename))
        image = image.convert("RGB")


        pixel_value = image.getpixel((pixel_x, pixel_y))


        return  {
            'pixel_value': pixel_value, 
            'camera_position': camera_position,
            'camera_orientation': camera_orientation,   
            'theta': theta, 
            'phi': phi,
        }
    

