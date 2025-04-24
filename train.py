"""Main training script for the model."""

import torch
from data import SyntheticDataset
from model import NeRFModel  
from utils import Camera, Volume
from utils import get_points


def nerf_l2_loss(preds, targets):
    """difference betwenn ground truth pixel value and 
    sum of predicted pixel values"""
    total_loss = 0.0
    for p, t in zip(preds, targets):
        total_loss += torch.mean((p - t) ** 2)
    return total_loss / len(preds)

def train(
    train_dataset_path: str = "",
    valid_dataset_path: str = "",
    test_dataset_path: str = "",
    run_name: str = "test",
    num_epochs: int = 10,
    num_layers: int = 3,
    num_hidden: int = 256,
    use_leaky_relu: bool = False,
    lr: float = 0.001,
    batch_size: int = 2,
) -> float:
    # Create a volume object
    volume = Volume()

    # Load the datasets
    train_dataset = SyntheticDataset(dataset_path=train_dataset_path)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = SyntheticDataset(dataset_path=valid_dataset_path)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Create the model
    model = NeRFModel(
        num_layers=num_layers,
        num_hidden=num_hidden,
        use_leaky_relu=use_leaky_relu,
    )

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Move the model to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Keep track of the best validation loss
    best_loss = float("inf")

    for epoch in range(num_epochs):
        # Training step
        model.train()
        train_loss = 0.0
        for d in train_dataloader:
            optimizer.zero_grad()

            # Move data to the same device as the model
            theta = d['theta'].unsqueeze(1).to(device)
            phi = d['phi'].unsqueeze(1).to(device)
            pixel_value = d['pixel_value'].to(device)

            # Create a camera object
            camera = Camera(
                position=d['camera_position'].to(device),
                orientation=d['camera_orientation'].to(device),
                focal_length=0.5,
                img_width=128,
                img_height=128,
            )

            # Get all points sampled in the ray
            predicted_pixel_value = torch.zeros(batch_size, 3).to(device)
            for point in get_points(camera, volume):
                # Combine theta, phi, and sampled point locations into an input tensor
                single_point_model_input = torch.cat((theta, phi, point), dim=1)

                # Forward pass
                outputs = model(single_point_model_input)

                # Compute the final pixel value for each output ray (ray marching)
                for b in range(batch_size):
                    opacity = outputs[b, 0]
                    color = outputs[b, 1:]
                    predicted_pixel_value[b] += (1 - opacity) * color

            # Compute loss
            loss = nerf_l2_loss(predicted_pixel_value, pixel_value)
            train_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

        train_loss /= len(train_dataloader)

        # Validation step
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for d in valid_dataloader:
                # Move data to the same device as the model
                theta = d['theta'].unsqueeze(1).to(device)
                phi = d['phi'].unsqueeze(1).to(device)
                pixel_value = d['pixel_value'].to(device)

                # Create a camera object
                camera = Camera(
                    position=d['camera_position'].to(device),
                    orientation=d['camera_orientation'].to(device),
                    focal_length=0.5,
                    img_width=128,
                    img_height=128,
                )

                # Get all points sampled in the ray
                predicted_pixel_value = torch.zeros(batch_size, 3).to(device)
                for point in get_points(camera, volume):
                    # Combine theta, phi, and sampled point locations into an input tensor
                    single_point_model_input = torch.cat((theta, phi, point), dim=1)

                    # Forward pass
                    outputs = model(single_point_model_input)

                    # Compute the final pixel value for each output ray (ray marching)
                    for b in range(batch_size):
                        opacity = outputs[b, 0]
                        color = outputs[b, 1:]
                        predicted_pixel_value[b] += (1 - opacity) * color

                # Compute loss
                loss = nerf_l2_loss(predicted_pixel_value, pixel_value)
                valid_loss += loss.item()

        valid_loss /= len(valid_dataloader)

        # Save the model checkpoint if validation loss improves
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), f"{run_name}_best_model.pth")

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}")

    return best_loss


# Add a test function to evaluate the model on the test dataset
def test(
    test_dataset_path: str = "",
    model_path: str = "test_best_model.pth",
    batch_size: int = 2,
) -> float:
    
    # Create a volume object
    volume = Volume()

    # Load the test dataset
    test_dataset = SyntheticDataset(dataset_path=test_dataset_path)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create the model
    model = NeRFModel()
    model.load_state_dict(torch.load(model_path))

    # Move the model to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Testing step
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for d in test_dataloader:
            # Move data to the same device as the model
            theta = d['theta'].unsqueeze(1).to(device)
            phi = d['phi'].unsqueeze(1).to(device)
            pixel_value = d['pixel_value'].to(device)

            # Create a camera object
            camera = Camera(
                position=d['camera_position'].to(device),
                orientation=d['camera_orientation'].to(device),
                focal_length=0.5,
                img_width=128,
                img_height=128,
            )

            # Get all points sampled in the ray
            predicted_pixel_value = torch.zeros(batch_size, 3).to(device)
            for point in get_points(camera, volume):
                # Combine theta, phi, and sampled point locations into an input tensor
                single_point_model_input = torch.cat((theta, phi, point), dim=1)

                # Forward pass
                outputs = model(single_point_model_input)

                # Compute the final pixel value for each output ray (ray marching)
                for b in range(batch_size):
                    opacity = outputs[b, 0]
                    color = outputs[b, 1:]
                    predicted_pixel_value[b] += (1 - opacity) * color

            # Compute loss
            loss = nerf_l2_loss(predicted_pixel_value, pixel_value)
            test_loss += loss.item()

    test_loss /= len(test_dataloader)
    print(f"Test Loss: {test_loss:.4f}")

if __name__ == "__main__":

# Debugging train and test
    train()
    test()

