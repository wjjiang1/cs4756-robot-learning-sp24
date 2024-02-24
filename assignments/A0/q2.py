import torch
import torch.nn as nn

def get_checkpoint_path():
    """Return the path to save the best performing model checkpoint.
    
    Returns:
        checkpoint_path (str)
            The path to save the best performing model checkpoint
    """
    return 'best_model_checkpoint.pth'

class LinearRegression(nn.Module):
    def __init__(self, input_dim=1, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
        

def create_loss_and_optimizer(model):
    """Create and return a loss function and optimizer.
    
    Parameters:
        model (torch.nn.Module)
            A neural network
        learning_rate (float)
            Learning rate for the optimizer
    
    Returns:
        loss_fn (function)
            The loss function for the model
        optimizer (torch.optim.Optimizer)
            The optimizer for the model
    """
    # TODO: Implement
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)
    return loss_fn, optimizer

def train(x, y, model, loss_fn, optimizer, checkpoint_path, num_epochs=1000):
    """Train a model.
    
    Parameters:
        x (torch.Tensor)
            The input data
        y (torch.Tensor)
            The expected output data
        model (torch.nn.Module)
            A neural network
        loss_fn (function)
            The loss function
        optimizer (torch.optim.Optimizer)
            The optimizer for the model
        checkpoint_path (str)
            The path to save the best performing checkpoint
        num_epochs (int)
            The number of epochs to train for
    
    Side Effects:
        - Save the best performing model checkpoint to `checkpoint_path`
    """
    # TODO: Implement
    model.train()
    best_loss = float('inf')
    
    for t in range(num_epochs):
        pred = model(x)
        loss = loss_fn(pred, y)
        if loss.item() < best_loss:
            torch.save(model.state_dict(), checkpoint_path)
            best_loss = loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def load_model_checkpoint(checkpoint_path):
    """Load a model checkpoint from disk.

    Parameters:
        checkpoint_path (str)
            The path to load the checkpoint from
    
    Returns:
        model (torch.nn.Module)
            The model loaded from the checkpoint
    """
    # TODO: Implement
    model = LinearRegression()
    model.load_state_dict(torch.load(checkpoint_path))
    return model
    