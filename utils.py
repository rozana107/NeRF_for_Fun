"""Ray and Volume utilities"""

import numpy as np
import torch
from dataclasses import dataclass

# the up direction in world coordinates
WORLD_UP_VEC = torch.tensor([0, 0, 1], dtype=torch.float32)

# A rya (vector) in world coordinates
@dataclass
class Ray:
    origin: torch.Tensor[3]
    direction: torch.Tensor[3]

# A Dataclass for a camera object
@dataclass
class Camera:
    position: torch.Tensor[3]
    orientation: torch.Tensor[3, 3]
    imae_width: int
    image_height: int
    focal_length: float

# A dtataclass for a volume object
@dataclass
class Volume:
    size: torch.Tensor[3]
    resolution: torch.Tensor[3]
    position: torch.Tensor[3] = torch.tensor([0, 0, 0], dtype=torch.float32)


def get_rays(
    camera_position,
    camera_orientation, 
    pixel_width, 
    pixel_height, 
    focal_length
    ):
    
    # calculate the camera's forward direction
    cam_forward = np.dot(camera_orientation, np.array([0, 0, -1]))

    # calculate the image plane's center point in world space
    image_center = camera_position + cam_forward * focal_length

    # calculate the image plane's width and height in world space
    image_width = 2 * focal_length / np.tan(np.radians(pixel_width / 2))
    image_height = 2 * focal_length / np.tan(np.radians(pixel_height / 2))   

    # calculate the image plane's right and up vectors in world space
    cam_right = np.cross(cam_forward, np.array([0, 1, 0]))
    cam_up = np.cross(cam_right, cam_forward)

    # calculate the image plane's top left corner in world space
    image_top_left = image_center + cam_up * (image_height / 2) - cam_right * (image_width / 2)

    # calculate the step sizes for the right and up directions
    right_step = cam_right * (image_width / pixel_width)
    up_step = cam_up * (image_height / pixel_height)

    # initialize the rays list to store the rays
    rays = []
    # loop through each pixel in the image plane
    for y in range(pixel_height):
        for x in range(pixel_width):
            # calculate the pixel's position in world space
            pixel_pos = image_top_left + x * right_step + y * up_step

            # calculate the ray's origin and direction
            # origin is the camera position
            # direction is the normalized vector from the camera to the pixel position
            ray_origin = camera_position   
            ray_direction = pixel_pos - camera_position        
            ray_direction = ray_direction / torch.norm(ray_direction)  # normalize the direction
            
            # add the ray to the list 
            rays.append(Ray(origin=ray_origin, direction=ray_direction))

    
    return rays

# Ray sampling

# Volume rendering
