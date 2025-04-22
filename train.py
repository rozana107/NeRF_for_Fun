"""Main training script for the model."""

import torch
from data import SyntheticDataset
from model import NeRFModel  

def load_dataset(dataset_path:str="data/ShapeNetCore.v2", batch_size:int=2):
    """
    Load the dataset from the specified path and return a DataLoader.
    
    Args:
        dataset_path (str): Path to the dataset.
        batch_size (int): Batch size for DataLoader.
    
    Returns:
        DataLoader: DataLoader for the dataset.
    """
    # Implement dataset loading logic here
    pass


def train(
    train_dataset_path:str="",
    valid_dataset_path:str="",
    test_dataset_path:str="",
    run_name:str="test",
    num_epochs:int=10,
    num_layers:int=3,
    num_hidden:int=256,
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
    )
    

    #Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #Create the loss function
    loss_fn = torch.nn.MSELoss()
    # Move the model to the GPU if available    
    if torch.cuda.is_available():
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    # Keep track of the best loss
    best_loss = float('inf')

    for epoch in range(num_epochs):
        # Iterate over the dataset
        for image, theta, phi in train_dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # TODO: Do Rendering 
            
            # Forward pass
            outputs = model(image, theta, phi)

            # Define target (e.g., ground truth image or expected output)
            target = image  # Replace this with the actual target from your dataset

            # Compute loss
            loss = loss_fn(outputs, target)

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

            # Compute loss
            loss = loss_fn(outputs, target)
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

        # Compute loss
        loss = loss_fn(outputs, target)
        test_loss += loss.item()
    test_loss /= len(test_dataloader)   

    
    return best_loss, test_loss



if __name__ == "__main__":

# Debugging train and test
    train()

    test()
