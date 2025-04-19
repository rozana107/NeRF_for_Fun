"""Ray and Volume utilities"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import List


# the up direction in world coordinates
WORLD_UP_VEC = torch.tensor([[0], [0], [1]], dtype=torch.float32)

# A rya (vector) in world coordinates
@dataclass
class Ray:
    """Dataclass for a ray object"""
    origin: torch.Tensor
    direction: torch.Tensor

@dataclass
class Camera:
    """Dataclass for a camera object"""
    position: torch.Tensor
    orientation: torch.Tensor  # 3x3 rotation matrix
    image_width: int
    image_height: int
    focal_length: float
    forward_direction_vec: torch.Tensor = torch.tensor([[0], [0], [-1]], dtype=torch.float32)
    right_direction_vec: torch.Tensor = torch.tensor([[0], [1], [0]], dtype=torch.float32)
    
@dataclass
class Volume:
    """Dataclass for a volume object"""
    size: torch.Tensor
    resolution: torch.Tensor
    position: torch.Tensor = torch.tensor([[0], [0], [0]], dtype=torch.float32)

    def normalize_position(self, position: torch.Tensor) -> torch.Tensor:
        """Normalize a position in world space to a position in the volume's coordinate system"""
        # Calculate the volume's center in world space
        volume_center = self.position + self.size / 2
        # Calculate the relative position of the point in the volume's coordinate system
        relative_position = position - volume_center
        # Normalize the position to the volume's coordinate system
        normalized_position = relative_position / self.size
        
        return normalized_position


class RayUtils:
    @staticmethod
    def get_rays(camera: Camera) -> List[Ray]:
        """Generate rays from the camera's position and orientation"""

        # calculate the camera's forward direction
        cam_forward = camera.orientation @ camera.forward_direction_vec

        # calculate the image plane's center point in world space
        image_center = camera.position + cam_forward * camera.focal_length

        # calculate the image plane's width and height in world space
        image_width = 2 * camera.focal_length * np.tan(np.radians(camera.image_width / 2))
        image_height = 2 * camera.focal_length * np.tan(np.radians(camera.image_height / 2))

        # calculate the image plane's right and up vectors in world space
        cam_right = torch.cross(cam_forward.squeeze(), camera.right_direction_vec.squeeze(), dim=0)
        cam_up = torch.cross(cam_right, cam_forward.squeeze(), dim=0)

        # calculate the image plane's top left corner in world space
        image_top_left = image_center + cam_up * (image_height / 2) - cam_right * (image_width / 2)

        # calculate the step sizes for the right and up directions
        right_step = cam_right * (image_width / camera.image_width)
        up_step = cam_up * (image_height / camera.image_height)

        # initialize the rays list to store the rays
        rays = []
        # loop through each pixel in the image plane
        for y in range(camera.image_height):
            for x in range(camera.image_width):
                # calculate the pixel's position in world space
                pixel_pos = image_top_left + x * right_step + y * up_step

                # calculate the ray's origin and direction
                ray_origin = camera.position
                ray_direction = pixel_pos - camera.position
                ray_direction /= torch.linalg.norm(ray_direction)  # normalize the direction

                # add the ray to the list
                rays.append(Ray(origin=ray_origin, direction=ray_direction))

        return rays

# Ray sampling

# Volume sampling

# Volume rendering

if __name__ == "__main__":
    # Debugging for the ray calculation
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    #create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_aspect('equal')

    #create a camera object
    camera = Camera(
        position= torch.tensor([[2.45], [-2], [1.45]], dtype=torch.float32),
        orientation=torch.tensor([
            [0.6601, -0.2126,  0.7205],
            [0.7512,  0.1868, -0.6331],
            [0.0000,  0.9591,  0.2830]], dtype=torch.float32),
        image_width=20,
        image_height=20,
        focal_length=0.5
    )
    # get the rays from the camera
    rays = RayUtils.get_rays(camera)

    # plot the rays as lines with the origin at (0, 0, 0)
    for ray in rays:
        # Squeeze the tensors to ensure they are 1D
        ray_origin = ray.origin.view(-1)  # Flatten to shape [3]
        ray_direction = ray.direction.view(-1)  # Flatten to shape [3]

        ax.plot(
            [ray_origin[0].item(), ray_origin[0].item() + ray_direction[0].item()],
            [ray_origin[1].item(), ray_origin[1].item() + ray_direction[1].item()],
            [ray_origin[2].item(), ray_origin[2].item() + ray_direction[2].item()],
            '-o'
        )
        
    # set the labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # set the title of the plot
    ax.set_title('Rays from Camera')

    # set the limits of the plot
    ax.set_xlim([-1, 1])  
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # show the plot
    plt.show()
