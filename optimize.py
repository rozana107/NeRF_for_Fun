"""Main training script for the model."""

import hyperopt
from hyperopt import fmin, tpe, hp

from optimize import perform_one_run


def objective(params):
    """Objective function to minimize."""
    # Unpack the parameters
    encoder = params['encoder']
    batch_size = params['batch_size']
    lr = params['lr']
    num_epochs = params['num_epochs']
    step_size = params['step_size']
    gamma = params['gamma']

    # Here you would typically train your model and return the validation loss
    # For demonstration purposes, we will just return a random number
    # In a real scenario, you would replace this with your training code
    loss = random.uniform(0, 1)
    
    return loss


# 
if __name__ == "__main__":

    # Define the search space
    search_space = {
        'encoder':hp.choice('encoder', ['resnet', 'vgg'], [
            'esmlv_t33_650M_UR90S_1',
            'esmlv_t33_650M_UR90S_5',
            'esm2_t33_650M_UR50D',
        ]),
        'batch_size': hp.choice('batch_size', [128, 256, 512]),
        'lr': hp.loguniform('lr', np.log(0.0001), np.log(0.01)),
        'num_epochs': 200,
        # Learning rate scheduler
        'step_size': 50,
        'gamma': 0.1,
    }

    # Run the optimization
    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=100,
    )
    # Print the best dataset found
    print(best)
