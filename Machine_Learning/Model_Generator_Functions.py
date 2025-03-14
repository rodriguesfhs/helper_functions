#!/usr/bin/env python
# coding: utf-8

# In[58]:


import torch
import torch.nn as nn

def model_creator(input_layer_units: int, 
                  hidden_layers: int, 
                  hidden_layers_units: int, 
                  activation_layers: str = "relu", 
                  output_layer_units: int):
    """
    Creates a PyTorch model dynamically using nn.Sequential.

    Args:
        input_layer_units (int): Number of units in the input layer.
        hidden_layers (int): Number of hidden layers.
        hidden_layers_units (int): Number of units in each hidden layer.
        activation_layers (str): Activation function to use (e.g., 'relu', 'sigmoid', 'tanh').
        output_layer_units (int): Number of units in the output layer.

    Returns:
        model (nn.Sequential): A PyTorch neural network model.

    Supported activation functions:
        nn.ReLU() -> 'relu'
        nn.LeakyReLU() -> 'leakyrelu'
        nn.Sigmoid() -> 'sigmoid'
        nn.Tanh() -> 'tanh'    

    Example of usage:

        input_layer_units = 2
        hidden_layers = 2
        hidden_layers_units = 16
        activation_layers = 'relu'
        output_layer_units = 1
    
        model = model_creator(input_layer_units, 
                              hidden_layers, 
                              hidden_layers_units, 
                              activation_layers, 
                              output_layer_units)
        print(model)

    Output:
    Sequential(
      (0): Linear(in_features=2, out_features=16, bias=True)
      (1): ReLU()
      (2): Linear(in_features=16, out_features=16, bias=True)
      (3): ReLU()
      (4): Linear(in_features=16, out_features=16, bias=True)
      (5): ReLU()
      (6): Linear(in_features=16, out_features=1, bias=True))
    """

    
   # Define the activation function (if provided)
    if activation_layers is not None:
        if activation_layers == 'relu':
            activation = nn.ReLU()
        elif activation_layers == 'leakyrelu':
            activation = nn.LeakyReLU()
        elif activation_layers == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation_layers == 'tanh':
            activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation_layers}")
    else:
        activation = None  # No activation function

    
    # Create a list to hold the layers
    layers = []

    # Input layer
    layers.append(nn.Linear(input_layer_units, hidden_layers_units))
    if activation is not None:
        layers.append(activation)

    # Hidden layers
    for _ in range(hidden_layers):
        layers.append(nn.Linear(hidden_layers_units, hidden_layers_units))
        if activation is not None:
            layers.append(activation)

    # Output layer
    layers.append(nn.Linear(hidden_layers_units, output_layer_units))

    # Create the model using nn.Sequential
    model = nn.Sequential(*layers)
    return model


# In[ ]:


import torch
import torch.nn as nn

def model_creator_V2(input_layer_units: int,
                     hidden_layers: int,
                     hidden_layers_units: int,
                     activation_layers: str = "relu",
                     output_layer_units: int = 1,
                     dropout_prob: float = None,
                     use_batch_norm: bool = False,):
    """
    Creates a PyTorch model dynamically using nn.Sequential. It incorporates the
    capacity to add Dropout and Batch Normalisation layers too.

    Args:
        input_layer_units (int): Number of units in the input layer.
        hidden_layers (int): Number of hidden layers.
        hidden_layers_units (int): Number of units in each hidden layer.
        activation_layers (str): Activation function to use (e.g., 'relu', 'sigmoid', 'tanh').
        output_layer_units (int): Number of units in the output layer.
        dropout_prob (float, optional): Dropout probability. If None, no dropout is added.
        use_batch_norm (bool): Whether to add batch normalization layers.

    Returns:
        model (nn.Sequential): A PyTorch neural network model.

    Supported activation functions:
        nn.ReLU() -> 'relu'
        nn.LeakyReLU() -> 'leakyrelu'
        nn.Sigmoid() -> 'sigmoid'
        nn.Tanh() -> 'tanh' 

    Example of Usage:
        1. Basic Model (No Dropout or Batch Norm)
            model = model_creator_V2(
                        input_layer_units=2,
                        hidden_layers=2,
                        hidden_layers_units=10,
                        activation_layers="relu",
                        output_layer_units=1,)
            print(model)

        Output:
            Sequential(
              (0): Linear(in_features=2, out_features=10, bias=True)
              (1): ReLU()
              (2): Linear(in_features=10, out_features=10, bias=True)
              (3): ReLU()
              (4): Linear(in_features=10, out_features=1, bias=True))


        2. Model with Dropout
            model = model_creator_V2(
                        input_layer_units=2,
                        hidden_layers=2,
                        hidden_layers_units=10,
                        activation_layers="relu",
                        output_layer_units=1,
                        dropout_prob=0.5,)
            print(model)
            
        Output:
            Sequential(
              (0): Linear(in_features=2, out_features=10, bias=True)
              (1): ReLU()
              (2): Dropout(p=0.5, inplace=False)
              (3): Linear(in_features=10, out_features=10, bias=True)
              (4): ReLU()
              (5): Dropout(p=0.5, inplace=False)
              (6): Linear(in_features=10, out_features=1, bias=True))

        3. Model with Batch Normalization
            model = model_creator_V2(
                        input_layer_units=2,
                        hidden_layers=2,
                        hidden_layers_units=10,
                        activation_layers="relu",
                        output_layer_units=1,
                        use_batch_norm=True,)
            print(model)

        Output:
            Sequential(
              (0): Linear(in_features=2, out_features=10, bias=True)
              (1): BatchNorm1d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU()
              (3): Linear(in_features=10, out_features=10, bias=True)
              (4): BatchNorm1d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): ReLU()
              (6): Linear(in_features=10, out_features=1, bias=True))

        4. Model with Dropout and Batch Normalization
            model = model_creator_V2(
                        input_layer_units=2,
                        hidden_layers=2,
                        hidden_layers_units=10,
                        activation_layers="relu",
                        output_layer_units=1,
                        dropout_prob=0.5,
                        use_batch_norm=True,)
            print(model)

        Output:
            Sequential(
              (0): Linear(in_features=2, out_features=10, bias=True)
              (1): BatchNorm1d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU()
              (3): Dropout(p=0.5, inplace=False)
              (4): Linear(in_features=10, out_features=10, bias=True)
              (5): BatchNorm1d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (6): ReLU()
              (7): Dropout(p=0.5, inplace=False)
              (8): Linear(in_features=10, out_features=1, bias=True))
            
    """
    # Define the activation function
    if activation_layers is not None:
        activation_layers = activation_layers.lower()
        if activation_layers == "relu":
            activation = nn.ReLU()
        elif activation_layers == "leakyrelu":
            activation = nn.LeakyReLU()
        elif activation_layers == "sigmoid":
            activation = nn.Sigmoid()
        elif activation_layers == "tanh":
            activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation_layers}")
    else:
        activation = None  # No activation function

    # Create a list to hold the layers
    layers = []

    # Input layer
    layers.append(nn.Linear(input_layer_units, hidden_layers_units))
    if use_batch_norm:
        layers.append(nn.BatchNorm1d(hidden_layers_units))
    if activation is not None:
        layers.append(activation)
    if dropout_prob is not None:
        layers.append(nn.Dropout(dropout_prob))

    # Hidden layers
    for _ in range(hidden_layers):
        layers.append(nn.Linear(hidden_layers_units, hidden_layers_units))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_layers_units))
        if activation is not None:
            layers.append(activation)
        if dropout_prob is not None:
            layers.append(nn.Dropout(dropout_prob))

    # Output layer
    layers.append(nn.Linear(hidden_layers_units, output_layer_units))

    # Create the model using nn.Sequential
    model = nn.Sequential(*layers)
    return model


# In[47]:


import torch.optim as optim

def optimizer_creator(model, optimizer_name, learning_rate, **kwargs):
    """
    Creates a PyTorch optimizer dynamically based on the provided parameters.

    Args:
        model (nn.Module): The neural network model whose parameters will be optimized.
        optimizer_name (str): Name of the optimizer (e.g., 'sgd', 'adam', 'rmsprop').
        learning_rate (float): Learning rate for the optimizer.
        **kwargs: Additional keyword arguments for the optimizer (e.g., momentum, weight_decay).

    Returns:
        optimizer: A PyTorch optimizer.

    Supported optimizers:
        optim.SGD() -> 'sgd'
        optim.Adam() -> 'adam'
        optim.RMSprop() -> 'rmsprop'
        optim.Adagrad() -> 'adagrad'
        optim.AdamW() -> 'adamw'

    Example usage:
        # Create the optimizer
        optimizer = optimizer_creator(model, optimizer_name='adam', learning_rate=0.001, weight_decay=0.0001)
        print(optimizer)
    """
    # Convert optimizer_name to lowercase for case-insensitive comparison
    optimizer_name = optimizer_name.lower()

    # Select the optimizer
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, **kwargs)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, **kwargs)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, **kwargs)
    elif optimizer_name == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, **kwargs)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    return optimizer


# In[49]:


import torch.nn as nn

def loss_function_creator(loss_name, **kwargs):
    """
    Creates a PyTorch loss function dynamically based on the provided parameters.

    Args:
        loss_name (str): Name of the loss function (e.g., 'mse', 'crossentropy', 'l1').
        **kwargs: Additional keyword arguments for the loss function (e.g., reduction, weight).

    Returns:
        loss_fn: A PyTorch loss function.

    Supported loss functions:
        nn.MSELoss() -> 'mse'
        nn.L1Loss() -> 'l1'
        nn.CrossEntropyLoss() -> 'crossentropy'
        nn.BCELoss() -> 'bce'
        nn.BCEWithLogitsLoss() -> 'bcewithlogits'
        nn.NLLLoss() -> 'nll'
        nn.HuberLoss() -> 'huber'
        

    Example usage:
        # Create a loss function
        loss_fn = loss_function_creator(loss_name='mse', 
                                        reduction='mean')
        print(loss_fn)
    """
    # Convert loss_name to lowercase for case-insensitive comparison
    loss_name = loss_name.lower()

    # Select the loss function
    if loss_name == 'mse':
        loss_fn = nn.MSELoss(**kwargs)
    elif loss_name == 'l1':
        loss_fn = nn.L1Loss(**kwargs)
    elif loss_name == 'crossentropy':
        loss_fn = nn.CrossEntropyLoss(**kwargs)
    elif loss_name == 'bce':
        loss_fn = nn.BCELoss(**kwargs)
    elif loss_name == 'bcewithlogits':
        loss_fn = nn.BCEWithLogitsLoss(**kwargs)
    elif loss_name == 'nll':
        loss_fn = nn.NLLLoss(**kwargs)
    elif loss_name == 'huber':
        loss_fn = nn.HuberLoss(**kwargs)
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")

    return loss_fn


# In[ ]:




