"""Main training script for the model."""

import torch
from data import SyntheticDataset
from model import NeRFModel  

def nerf_l2_loss(preds, targets):
    """difference betwenn ground truth pixel value and 
    sum of predicted pixel values"""
    total_loss = 0.0
    for p, t in zip(preds, targets):
        total_loss += torch.mean((p - t) ** 2)
    return total_loss / len(preds)

def train(
    train_dataset_path:str="",
    valid_dataset_path:str="",
    test_dataset_path:str="",
    run_name:str="test",
    num_epochs:int=10,
    num_layers:int=3,
    num_hidden:int=256,
    use_leaky_relu:bool=False,
    lr:float=0.001,
    batch_size:int=2,
) -> float:
    # Load the dataset
    train_dataset = SyntheticDataset(dataset_path=train_dataset_path)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Load the test dataset
    test_dataset = SyntheticDataset(dataset_path=test_dataset_path)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Load the validation dataset
    valid_dataset = SyntheticDataset(dataset_path=valid_dataset_path)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False) 


    #Create the model
    model = NeRFModel(
        num_layers = num_layers,
        num_hidden = num_hidden,
        use_leaky_relu=use_leaky_relu,
    )
    

    #  Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Move the model to the GPU if available    
    if torch.cuda.is_available():
        model = model.cuda()
        nerf_l2_loss = nerf_l2_loss.cuda()

    # Keep track of the best loss
    best_loss = float('inf')

    for epoch in range(num_epochs):
        # Iterate over the dataset
        for image, theta, phi in train_dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # TODO: Do Rendering 
            # Combine the theta nad phi and sample point locations into a innput tensor
            image = torch.cat((theta, phi), dim=1)
            image = image.view(-1, 3)  # Flatten the image tensor if needed   

            # Forward pass
            outputs = model(image, theta, phi)

            # Define target (e.g., ground truth image or expected output)
            target = image  # Replace this with the actual target from your dataset

            # Compute loss
            loss = nerf_l2_loss(
                # A pixel color from a single ray
                outputs, 
                # A pixel color in the ground truth image
                target
            )

            # Check to see if the loss is the best loss so far
            if loss < best_loss:
                best_loss = loss.item()

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

        # validation step
        valid_loss = 0.0
        for image, theta, phi in valid_dataloader:
            # Forward pass
            outputs = model(image, theta, phi)

            target = image

            # Compute loss
            loss = nerf_l2_loss(outputs, target)
            valid_loss += loss.item()
        valid_loss /= len(valid_dataloader) 

        # Save the model checkpoint
        if valid_loss < best_loss:
            best_loss = valid_loss
            # Save the model state here, e.g., torch.save(model.state_dict(), f"{run_name}_best_model.pth")
            torch.save(model.state_dict(), f"{run_name}_best_model.pth")
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Validation Loss: {valid_loss}")
    
    # Test
    test_loss = 0.0
    for image, theta, phi in test_dataloader:
        # Forward pass
        outputs = model(image, theta, phi)

        target = image

        # Compute loss
        loss = nerf_l2_loss(outputs, target)
        test_loss += loss.item()
    test_loss /= len(test_dataloader)   

    

    print(f"Test Loss: {test_loss}")

    return best_loss, test_loss



if __name__ == "__main__":

# Debugging train and test
    train()

