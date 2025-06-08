#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import essential libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch
import torch.optim as optim 
import torch.nn as nn

# Import metrics libraries
import torchmetrics
from torchmetrics import Accuracy
from torchmetrics import ConfusionMatrix
import mlxtend
from mlxtend.plotting import plot_confusion_matrix
from mlxtend.evaluate import confusion_matrix
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard

from tqdm.auto import tqdm


# In[3]:


# Trainer function for PyTorch

def train(model: torch.nn.Module,
           model_name: str,
           train_loader,
           test_loader,
           optimizer_class: type[optim.Optimizer] = optim.Adam,
           label_smoothing: float = 0.0,
           epochs: int = 20, 
           lr: float = 1e-3, 
           weight_decay=1e-4,
           epoch_tick: int =10,
           device='cpu'):

    """ This function is a template for training a PyTorch model.
    It takes a PyTorch model, a dataset, and an optimizer as input.
        Args:
        model (torch.nn.Module): The PyTorch model to train.
        model_name (str): The name of the model.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        test_loader (torch.utils.data.DataLoader): The test data loader.
        optimizer_class (type[optim.Optimizer]): The optimizer class to use.
        epochs (int): The number of epochs to train for.
        lr (float): The learning rate to use.
        weight_decay (float): The weight decay to use.
        epoch_tick (int): The number of epochs to wait before printing metrics.
        device (str): The device to train on.
    """
    

    model = model.to(device)
    MODEL_NAME = model_name
    optimizer = optimizer_class(params=model.parameters(),
                                lr=lr,
                                weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Store metrics
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # Initialize torchmetrics for evaluation
    metric_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=3).to(device)
    metric_f1 = torchmetrics.F1Score(task='multiclass', num_classes=3, average='macro').to(device)
    metric_precision = torchmetrics.Precision(task='multiclass', num_classes=3, average='macro').to(device)
    metric_recall = torchmetrics.Recall(task='multiclass', num_classes=3, average='macro').to(device)


    # TensorBoard
    writer = SummaryWriter(log_dir=f'runs/{MODEL_NAME}')
    sample_batch = next(iter(train_loader))[0].to(device)
    writer.add_graph(model, sample_batch)

    torch.manual_seed(42)

    for epoch in tqdm(range(1, epochs + 1)):

        all_test_preds = []
        all_test_labels = []

        # --- Training ---
        model.train()
        train_loss = 0
        train_correct = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)
            train_correct += (preds.argmax(dim=1) == y_batch).sum().item()

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)

        # --- Evaluation ---
        model.eval()
        test_loss = 0
        test_correct = 0

        # Reset metrics
        metric_accuracy.reset()
        metric_precision.reset()
        metric_recall.reset()
        metric_f1.reset()

        with torch.inference_mode():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = model(x_batch)
                loss = loss_fn(preds, y_batch)

                test_loss += loss.item() * x_batch.size(0)
                test_correct += (preds.argmax(dim=1) == y_batch).sum().item()

                pred_labels = preds.argmax(dim=1)
                all_test_preds.extend(pred_labels.cpu().numpy())
                all_test_labels.extend(y_batch.cpu().numpy())

                # Update metrics
                metric_accuracy.update(pred_labels, y_batch)
                metric_precision.update(pred_labels, y_batch)
                metric_recall.update(pred_labels, y_batch)
                metric_f1.update(pred_labels, y_batch)


        avg_test_loss = test_loss / len(test_loader.dataset)
        test_acc = test_correct / len(test_loader.dataset)

        # Compute final metric values
        acc_score = metric_accuracy.compute().item()
        prec_score = metric_precision.compute().item()
        recall_score = metric_recall.compute().item()
        f1_score = metric_f1.compute().item()

        # Store metrics
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        if epoch % epoch_tick == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:03d} | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.4f} | "
                  f"Prec: {prec_score:.4f} | Recall: {recall_score:.4f} | ")
            print(f"F1: {f1_score:.4f} |")

        # Write to TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/test', avg_test_loss, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        writer.add_scalar('Precision/test', prec_score, epoch)
        writer.add_scalar('Recall/test', recall_score, epoch)
        writer.add_scalar('F1Score/test', f1_score, epoch)


    # --- Plotting ---
    epochs_range = range(1, epochs + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, test_losses, label='Test Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Loss over Epochs")
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
    plt.plot(epochs_range, test_accuracies, label='Test Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} - Accuracy over Epochs")
    plt.legend()
    
    plt.tight_layout()
    plt.show()


    # --- Final Confusion Matrix ---
    label_map = {0: 'P', 1: 'H', 2: 'N'}
    class_names = [label_map[i] for i in sorted(set(all_test_labels))]

    # class_names = sorted(list(set(all_test_labels)))
    cm = confusion_matrix(all_test_labels, all_test_preds)
    
    fig_cm, ax = plot_confusion_matrix(conf_mat=cm,
                                   class_names=class_names,
                                   show_normed=True,
                                   colorbar=True,
                                   cmap='Blues',
                                   figsize=(4, 4))  # Pass figsize here

    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    
    writer.add_figure("ConfusionMatrix/test", fig_cm, global_step=epochs)
    # print("Saving local copy of confusion matrix...")
    # fig_cm.savefig("debug_conf_matrix.png")
    plt.show()
    plt.close(fig_cm)

    writer.close()
    print(f"TensorBoard logs saved to: runs/{MODEL_NAME}")
    print("To view TensorBoard, run the following command in your terminal:")
    print("tensorboard --logdir=runs")



# In[20]:


def generic_plotter(data_list, labels, colors, title="Plot"):
    """
    Plots multiple line plots on a single figure (simplified version).

    Args:
        data_list (list of Pandas DataFrames): A list of DataFrames, each with 'Step' and 'Value' columns.
        labels (list of str): A list of labels for each DataFrame.
        colors (list of str): A list of colors for each DataFrame.
        title (str, optional): The title of the plot. Defaults to "Plot".
    """
    plt.figure(figsize=(10, 4))
    for data, label, color in zip(data_list, labels, colors):
        sns.lineplot(data=data, x='Step', y=data['Value'], label=label, color=color,alpha=0.2)
        sns.lineplot(data=data, x='Step', y=data['Value'].ewm(alpha=0.1, adjust=False).mean(), label=label, color=color)

    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


# In[23]:


def flexible_plotter(data_list, labels, colors, title="Plot",
                     fontsize:int = 10,
                     SMOOTHING: float = 0.1):
    """
    Plots multiple line plots on a single figure (simplified version).

    Args:
        data_list (list of Pandas DataFrames): A list of DataFrames, each with 'Step' and 'Value' columns.
        labels (list of str): A list of labels for each DataFrame.
        colors (list of str): A list of colors for each DataFrame.
        title (str, optional): The title of the plot. Defaults to "Plot".

        Example:

            train_losses = [df16_train_loss,
                            df17_train_loss,
                            df18_train_loss]
                            
            model_names = ['Model 16', 'Model 17', 'Model 18']
            model_colors = ['#D33B90', '#EDAF3D', '#883ADE']

            flexible_plotter(train_losses, model_names, model_colors, title="Comparison of Train Loss")
    """
    ALPHA=0.2
    plt.figure(figsize=(10, 4), dpi=150)
    for data, label, color in zip(data_list, labels, colors):
        sns.lineplot(data=data, x='Step', y=data['Value'],  color=color, alpha=0.2)
        sns.lineplot(data=data, x='Step', y=data['Value'].ewm(alpha=SMOOTHING, adjust=False).mean(), 
                     color=color,label=label)

    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend(fontsize=fontsize)
    plt.show()


# In[26]:


def extract_scalars(event_file, tags=None):
    ea = EventAccumulator(event_file)
    ea.Reload()
    
    scalar_data = {}
    for tag in tags or ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        df = pd.DataFrame({"Step": steps, "Value": values})
        scalar_data[tag] = df
    return scalar_data

def process_all_runs(base_path):
    all_metrics = {}
    model_dirs = glob.glob(os.path.join(base_path, 'cnn_wider_v*'))

    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        event_files = glob.glob(os.path.join(model_dir, "events.out.tfevents.*"))

        if not event_files:
            continue

        print(f"Processing: {model_name}")
        scalar_dict = extract_scalars(event_files[0])

        # New tag map based on actual scalar tags
        tag_map = {
            'Loss/train': 'train_loss',
            'Loss/test': 'test_loss',
            'Accuracy/train': 'train_acc',
            'Accuracy/test': 'test_acc',
            'Precision/test': 'test_precision',
            'Recall/test': 'test_recall',
            'F1Score/test': 'test_f1score'
        }

        metrics = {}

        for tag, key in tag_map.items():
            if tag in scalar_dict:
                df = scalar_dict[tag]
                metrics[key] = df

                # Save as CSV
                csv_path = os.path.join(model_dir, f"{model_name}_{key}.csv")
                df.to_csv(csv_path, index=False)

        all_metrics[model_name] = metrics

    return all_metrics


# In[29]:


def load_metrics(model_num):
    prefix = f'cnn_avg_pool_v{model_num}'
    train_loss_path = f'{base_path}/{prefix}/{prefix}_train_loss.csv'
    test_loss_path = f'{base_path}/{prefix}/{prefix}_test_loss.csv'
    train_acc_path = f'{base_path}/{prefix}/{prefix}_train_acc.csv'
    test_acc_path = f'{base_path}/{prefix}/{prefix}_test_acc.csv'
    return {
        'train_loss': pd.read_csv(train_loss_path, header=[0]),
        'test_loss': pd.read_csv(test_loss_path, header=[0]),
        'train_acc': pd.read_csv(train_acc_path, header=[0]),
        'test_acc': pd.read_csv(test_acc_path, header=[0])
    }


# In[ ]:


def multi_plotter(data_dict: dict,
                  ncols: int = 1,
                  fontsize: int = 10,
                  SMOOTHING: float = 0.1):
    """
    Plots train/test loss and accuracy for multiple models with smoothing.
    """
    plt.figure(figsize=(14, 9), dpi=100)

    for model_name, model_data in data_dict.items():
        color = model_data.get('color')
        ALPHA = 0.2

        # Train Loss
        plt.subplot(3, 2, 1)        
        # Raw train loss (semi-transparent)
        plt.plot(model_data['train_loss']['Step'],
                 model_data['train_loss']['Value'],
                 color=color, alpha=ALPHA)
        
        # Smoothed train loss (with label)
        smoothed = model_data['train_loss']['Value'].ewm(alpha=SMOOTHING, adjust=False).mean()
        plt.plot(model_data['train_loss']['Step'],
                 smoothed,
                 label=model_name,
                 color=color)
        
        plt.title("Train loss")
        plt.xlabel("Step (Epoch)")
        plt.ylabel("Loss")
        plt.legend(loc='best', ncols=ncols, fontsize=fontsize)


        # Test Loss
        plt.subplot(3, 2, 2)        
        # Raw test loss (semi-transparent)
        plt.plot(model_data['test_loss']['Step'],
                 model_data['test_loss']['Value'],
                 color=color, alpha=ALPHA)
        
        # Smoothed test loss (with label)
        smoothed = model_data['test_loss']['Value'].ewm(alpha=SMOOTHING, adjust=False).mean()
        plt.plot(model_data['test_loss']['Step'],
                 smoothed,
                 label=model_name,
                 color=color)
        
        plt.title("Test loss")
        plt.xlabel("Step (Epoch)")
        plt.ylabel("Loss")
        plt.legend(loc='best', ncols=ncols, fontsize=fontsize)


        # Train Accuracy
        plt.subplot(3, 2, 3)        
        # Raw train accuracy (semi-transparent)
        plt.plot(model_data['train_acc']['Step'],
                 model_data['train_acc']['Value'],
                 color=color, alpha=ALPHA)
        
        # Smoothed train accuracy (with label)
        smoothed = model_data['train_acc']['Value'].ewm(alpha=SMOOTHING, adjust=False).mean()
        plt.plot(model_data['train_acc']['Step'],
                 smoothed,
                 label=model_name,
                 color=color)
        
        plt.title("Train accuracy")
        plt.xlabel("Step (Epoch)")
        plt.ylabel("Accuracy")
        plt.legend(loc='best', ncols=ncols, fontsize=fontsize)


        # Test Accuracy
        plt.subplot(3, 2, 4)        
        # Raw test accuracy (semi-transparent)
        plt.plot(model_data['test_acc']['Step'],
                 model_data['test_acc']['Value'],
                 color=color, alpha=ALPHA)
        
        # Smoothed test accuracy (with label)
        smoothed = model_data['test_acc']['Value'].ewm(alpha=SMOOTHING, adjust=False).mean()
        plt.plot(model_data['test_acc']['Step'],
                 smoothed,
                 label=model_name,
                 color=color)
        
        plt.title("Test accuracy")
        plt.xlabel("Step (Epoch)")
        plt.ylabel("Accuracy")
        plt.legend(loc='best', ncols=ncols, fontsize=fontsize)


        # Train/Test Loss
        plt.subplot(3, 2, 5)
        
        # Raw train (semi-transparent)
        plt.plot(model_data['train_loss']['Step'],
                 model_data['train_loss']['Value'],
                 color=color, alpha=ALPHA,
                 label=None)
        
        # Smoothed train (with label)
        smoothed_train = model_data['train_loss']['Value'].ewm(alpha=SMOOTHING, adjust=False).mean()
        plt.plot(model_data['train_loss']['Step'],
                 smoothed_train,
                 label=f"{model_name}",
                 color=color)
        
        # Raw test (semi-transparent)
        plt.plot(model_data['test_loss']['Step'],
                 model_data['test_loss']['Value'],
                 color=color, alpha=ALPHA,
                 label=None,
                 ls='--')
        
        # Smoothed test (with label)
        smoothed_test = model_data['test_loss']['Value'].ewm(alpha=SMOOTHING, adjust=False).mean()
        plt.plot(model_data['test_loss']['Step'],
                 smoothed_test,
                 # label=f"Test {model_name}",
                 color=color,
                 label=None,
                 ls='--')
        
        plt.title("Train (solid) / Test (dashed) loss")
        plt.xlabel("Step (Epoch)")
        plt.ylabel("Loss")
        plt.legend(loc='best', ncols=ncols, fontsize=fontsize)
    

        # Train/Test Accuracy
        plt.subplot(3, 2, 6)
        
        # Raw train accuracy (semi-transparent)
        plt.plot(model_data['train_acc']['Step'],
                 model_data['train_acc']['Value'],
                 color=color, alpha=ALPHA,
                 label=None)
        
        # Smoothed train accuracy (with label)
        smoothed_train = model_data['train_acc']['Value'].ewm(alpha=SMOOTHING, adjust=False).mean()
        plt.plot(model_data['train_acc']['Step'],
                 smoothed_train,
                 label=f"{model_name}",
                 color=color)
        
        # Raw test accuracy (semi-transparent)
        plt.plot(model_data['test_acc']['Step'],
                 model_data['test_acc']['Value'],
                 color=color, alpha=ALPHA,
                 label=None,
                 ls='--')
        
        # Smoothed test accuracy (with label)
        smoothed_test = model_data['test_acc']['Value'].ewm(alpha=SMOOTHING, adjust=False).mean()
        plt.plot(model_data['test_acc']['Step'],
                 smoothed_test,
                 # label=f"Test {model_name}",
                 color=color,
                 label=None,
                 ls='--')
        
        plt.title("Train (solid) / Test (dashed) accuracy ")
        plt.xlabel("Step (Epoch)")
        plt.ylabel("Accuracy")
        plt.legend(loc='best', ncols=ncols, fontsize=fontsize)

        plt.tight_layout()
    plt.show()


# In[ ]:


def multi_plotter2(data_dict: dict,
                  ncols: int = 1,
                  fontsize: int = 10,
                  SMOOTHING: float = 0.1):
    """
    Plots train/test loss and accuracy for multiple models with smoothing.
    """
    plt.figure(figsize=(14, 9), dpi=100)

    for model_name, model_data in data_dict.items():
        color = model_data.get('color')
        ALPHA = 0.2

        # Train Loss
        plt.subplot(3, 2, 1)        
        # Raw train loss (semi-transparent)
        plt.plot(model_data['train_loss']['Step'],
                 model_data['train_loss']['Value'],
                 color=color, alpha=ALPHA)
        
        # Smoothed train loss (with label)
        smoothed = model_data['train_loss']['Value'].ewm(alpha=SMOOTHING, adjust=False).mean()
        plt.plot(model_data['train_loss']['Step'],
                 smoothed,
                 label=model_name,
                 color=color)
        
        plt.title("Train loss")
        plt.xlabel("Step (Epoch)")
        plt.ylabel("Loss")
        plt.legend(loc='best', ncols=ncols, fontsize=fontsize)


        # Test Loss
        plt.subplot(3, 2, 2)        
        # Raw test loss (semi-transparent)
        plt.plot(model_data['test_loss']['Step'],
                 model_data['test_loss']['Value'],
                 color=color, alpha=ALPHA)
        
        # Smoothed test loss (with label)
        smoothed = model_data['test_loss']['Value'].ewm(alpha=SMOOTHING, adjust=False).mean()
        plt.plot(model_data['test_loss']['Step'],
                 smoothed,
                 label=model_name,
                 color=color)
        
        plt.title("Test loss")
        plt.xlabel("Step (Epoch)")
        plt.ylabel("Loss")
        plt.legend(loc='best', ncols=ncols, fontsize=fontsize)


        # Train Accuracy
        plt.subplot(3, 2, 3)        
        # Raw train accuracy (semi-transparent)
        plt.plot(model_data['train_acc']['Step'],
                 model_data['train_acc']['Value'],
                 color=color, alpha=ALPHA)
        
        # Smoothed train accuracy (with label)
        smoothed = model_data['train_acc']['Value'].ewm(alpha=SMOOTHING, adjust=False).mean()
        plt.plot(model_data['train_acc']['Step'],
                 smoothed,
                 label=model_name,
                 color=color)
        
        plt.title("Train accuracy")
        plt.xlabel("Step (Epoch)")
        plt.ylabel("Accuracy")
        plt.legend(loc='best', ncols=ncols, fontsize=fontsize)


        # Test Accuracy
        plt.subplot(3, 2, 4)        
        # Raw test accuracy (semi-transparent)
        plt.plot(model_data['test_acc']['Step'],
                 model_data['test_acc']['Value'],
                 color=color, alpha=ALPHA)
        
        # Smoothed test accuracy (with label)
        smoothed = model_data['test_acc']['Value'].ewm(alpha=SMOOTHING, adjust=False).mean()
        plt.plot(model_data['test_acc']['Step'],
                 smoothed,
                 label=model_name,
                 color=color)
        
        plt.title("Test accuracy")
        plt.xlabel("Step (Epoch)")
        plt.ylabel("Accuracy")
        plt.legend(loc='best', ncols=ncols, fontsize=fontsize)


        # Train/Test Loss
        plt.subplot(3, 2, 5)
        
        # Raw train (semi-transparent)
        plt.plot(model_data['train_loss']['Step'],
                 model_data['train_loss']['Value'],
                 color=color, alpha=ALPHA,
                 label=None)
        
        # Smoothed train (with label)
        smoothed_train = model_data['train_loss']['Value'].ewm(alpha=SMOOTHING, adjust=False).mean()
        plt.plot(model_data['train_loss']['Step'],
                 smoothed_train,
                 label=f"Train {model_name}",
                 color=color)
        
        # Raw test (semi-transparent)
        plt.plot(model_data['test_loss']['Step'],
                 model_data['test_loss']['Value'],
                 color=color, alpha=ALPHA,
                 label=None)
        
        # Smoothed test (with label)
        smoothed_test = model_data['test_loss']['Value'].ewm(alpha=SMOOTHING, adjust=False).mean()
        plt.plot(model_data['test_loss']['Step'],
                 smoothed_test,
                 label=f"Test {model_name}",
                 color=color,
                 ls='--')
        
        plt.title("Train/Test loss")
        plt.xlabel("Step (Epoch)")
        plt.ylabel("Loss")
        plt.legend(loc='best', ncols=ncols, fontsize=fontsize)
    

        # Train/Test Accuracy
        plt.subplot(3, 2, 6)
        
        # Raw train accuracy (semi-transparent)
        plt.plot(model_data['train_acc']['Step'],
                 model_data['train_acc']['Value'],
                 color=color, alpha=ALPHA,
                 label=None)
        
        # Smoothed train accuracy (with label)
        smoothed_train = model_data['train_acc']['Value'].ewm(alpha=SMOOTHING, adjust=False).mean()
        plt.plot(model_data['train_acc']['Step'],
                 smoothed_train,
                 label=f"Train {model_name}",
                 color=color)
        
        # Raw test accuracy (semi-transparent)
        plt.plot(model_data['test_acc']['Step'],
                 model_data['test_acc']['Value'],
                 color=color, alpha=ALPHA,
                 label=None
                )
        
        # Smoothed test accuracy (with label)
        smoothed_test = model_data['test_acc']['Value'].ewm(alpha=SMOOTHING, adjust=False).mean()
        plt.plot(model_data['test_acc']['Step'],
                 smoothed_test,
                 label=f"Test {model_name}",
                 color=color,
                 ls='--')
        
        plt.title("Train/Test accuracy")
        plt.xlabel("Step (Epoch)")
        plt.ylabel("Accuracy")
        plt.legend(loc='best', ncols=ncols, fontsize=fontsize)

    plt.tight_layout()
    plt.show()


# In[ ]:


# def multi_plotter(data_dict: dict,
#                   ncols: int = 1,
#                   fontsize: int = 10,
#                   SMOOTHING: float = 0.1):
#     """
#     Plots train/test loss and accuracy for multiple models with smoothing.
#     """
#     plt.figure(figsize=(14, 9), dpi=100)

#     for model_name, model_data in data_dict.items():
#         color = model_data.get('color')
#         ALPHA = 0.2

#         # Train Loss
#         plt.subplot(3, 2, 1)
#         sns.lineplot(data=model_data['train_loss'], x='Step', y='Value', color=color, alpha=ALPHA)
#         smoothed = model_data['train_loss']['Value'].ewm(alpha=SMOOTHING, adjust=False).mean()
#         sns.lineplot(x=model_data['train_loss']['Step'], y=smoothed, 
#                      label=model_name, 
#                      color=color)
#         plt.title("Train loss")
#         plt.xlabel("Step (Epoch)")
#         plt.ylabel("Loss")

#         # Test Loss
#         plt.subplot(3, 2, 2)
#         sns.lineplot(data=model_data['test_loss'], x='Step', y='Value', color=color, alpha=ALPHA)
#         smoothed = model_data['test_loss']['Value'].ewm(alpha=SMOOTHING, adjust=False).mean()
#         sns.lineplot(x=model_data['test_loss']['Step'], y=smoothed, 
#                      label=model_name, 
#                      color=color)
#         plt.title("Test loss")
#         plt.xlabel("Step (Epoch)")
#         plt.ylabel("Loss")

#         # Train Accuracy
#         plt.subplot(3, 2, 3)
#         sns.lineplot(data=model_data['train_acc'], x='Step', y='Value', color=color, alpha=ALPHA)
#         smoothed = model_data['train_acc']['Value'].ewm(alpha=SMOOTHING, adjust=False).mean()
#         sns.lineplot(x=model_data['train_acc']['Step'], y=smoothed, 
#                      label=model_name, 
#                      color=color)
#         plt.title("Train accuracy")
#         plt.xlabel("Step (Epoch)")
#         plt.ylabel("Accuracy")

#         # Test Accuracy
#         plt.subplot(3, 2, 4)
#         sns.lineplot(data=model_data['test_acc'], x='Step', y='Value', color=color, alpha=ALPHA)
#         smoothed = model_data['test_acc']['Value'].ewm(alpha=SMOOTHING, adjust=False).mean()
#         sns.lineplot(x=model_data['test_acc']['Step'], y=smoothed, 
#                      label=model_name, 
#                      color=color)
#         plt.title("Test accuracy")
#         plt.xlabel("Step (Epoch)")
#         plt.ylabel("Accuracy")

#         # # Train/Test Loss
#         plt.subplot(3, 2, 5)
#         sns.lineplot(data=model_data['train_loss'], x='Step', y='Value', color=color, alpha=ALPHA,label=None)
#         smoothed_train = model_data['train_loss']['Value'].ewm(alpha=SMOOTHING, adjust=False).mean()
#         sns.lineplot(x=model_data['train_loss']['Step'], y=smoothed_train, 
#                      label=f"Train {model_name}", 
#                      color=color)
#         sns.lineplot(data=model_data['test_loss'], x='Step', y='Value', color=color, alpha=ALPHA,label=None)
#         smoothed_test = model_data['test_loss']['Value'].ewm(alpha=SMOOTHING, adjust=False).mean()
#         sns.lineplot(x=model_data['test_loss']['Step'], y=smoothed_test, 
#                      label=f"Test {model_name}", 
#                      color=color)
#         plt.title("Train/Test loss")
#         plt.xlabel("Step (Epoch)")
#         plt.ylabel("Loss")

#         # # Train/Test Accuracy
#         plt.subplot(3, 2, 6)
#         sns.lineplot(data=model_data['train_acc'], x='Step', y='Value', color=color, alpha=ALPHA,label=None)
#         smoothed_train = model_data['train_acc']['Value'].ewm(alpha=SMOOTHING, adjust=False).mean()
#         sns.lineplot(x=model_data['train_acc']['Step'], y=smoothed_train, 
#                      label=f"Train {model_name}", 
#                      color=color)
#         sns.lineplot(data=model_data['test_acc'], x='Step', y='Value', color=color, alpha=ALPHA,label=None)
#         smoothed_test = model_data['test_acc']['Value'].ewm(alpha=SMOOTHING, adjust=False).mean()
#         sns.lineplot(x=model_data['test_acc']['Step'], y=smoothed_test, 
#                      label=f"Test {model_name}", 
#                      color=color)
#         plt.title("Train/Test accuracy")
#         plt.xlabel("Step (Epoch)")
#         plt.ylabel("Accuracy")
#         
#     plt.tight_layout()
#     plt.legend(loc='best', ncols=ncols, fontsize=fontsize)
#     plt.show()


# In[ ]:


# def multi_plotter2(data_dict: dict,
#                   ncols:int = 1,
#                   fontsize:int =10,
#                   SMOOTHING: float=0.1):
#     """
#     Generates a 2x2 figure with subplots for train loss, test loss,
#     train accuracy, and test accuracy for multiple models, with colors
#     specified inline within the data dictionary.

#     Args:
#         data_dict (dict): A dictionary where keys are model names (str)
#                            and values are dictionaries containing the DataFrames
#                            for 'train_loss', 'test_loss', 'train_acc', 'test_acc',
#                            and the color associated with the model name (e.g., 'Model 16': '#D33B90').
#                            Each of the DataFrames MUST have 'Step' and 'Value' columns.
#         SMOOTHING (float): Pandas ewm (Exponentially Weighted Moving) smoothing. The higher the value, the lower
#                             the smoothing is. It adjusts ewm alpha parameter.
#                            Example:
#                            {
#                                'Model 16': {'train_loss': df16_train_loss, 
#                                             'test_loss': df16_test_loss, 
#                                             'train_acc': df16_train_acc, 
#                                             'test_acc': df16_test_acc, 
#                                             'color': '#D33B90'},
                                            
#                                'Model 17': {'train_loss': df17_train_loss, 
#                                             'test_loss': df17_test_loss, 
#                                             'train_acc': df17_train_acc, 
#                                             'test_acc': df17_test_acc, 
#                                             'color': '#EDAF3D'},
                                            
#                                'Model 18': {'train_loss': df18_train_loss, 
#                                             'test_loss': df18_test_loss, 
#                                             'train_acc': df18_train_acc, 
#                                             'test_acc': df18_test_acc, 
#                                             'color': '#883ADE'}
#                            }
#     """
#     plt.figure(figsize=(14, 9), dpi=100)

#     for model_name, model_data in data_dict.items():
#         color = model_data.get('color')
        
#         ALPHA=0.2
#         # SMOOTHING=0.1
#         # Train Loss
#         plt.subplot(2, 2, 1)
#         sns.lineplot(data=model_data['train_loss'], x='Step', y='Value', 
#                      color=color,
#                      alpha=ALPHA)
#         sns.lineplot(data=model_data['train_loss'], x='Step', y=model_data['train_loss']['Value'].ewm(alpha=SMOOTHING, 
#                                                                                                       adjust=False).mean(), 
#                      label=model_name, 
#                      color=color)
#         plt.title("Train loss")
#         plt.xlabel("Step (Epoch)")
#         plt.ylabel("Loss")

#         # Test Loss
#         plt.subplot(2, 2, 2)
#         sns.lineplot(data=model_data['test_loss'], x='Step', y='Value', 
#                      color=color,
#                      alpha=ALPHA)
#         sns.lineplot(data=model_data['test_loss'], x='Step', y=model_data['test_loss']['Value'].ewm(alpha=SMOOTHING, 
#                                                                                                       adjust=False).mean(), 
#                      label=model_name, 
#                      color=color)
#         plt.title("Test loss")
#         plt.xlabel("Step (Epoch)")
#         plt.ylabel("Loss")

#         # Train Accuracy
#         plt.subplot(2, 2, 3)
#         sns.lineplot(data=model_data['train_acc'], x='Step', y='Value', 
#                      color=color,
#                      alpha=ALPHA)
#         sns.lineplot(data=model_data['train_acc'], x='Step', y=model_data['train_acc']['Value'].ewm(alpha=SMOOTHING, 
#                                                                                                       adjust=False).mean(), 
#                      label=model_name, 
#                      color=color)
#         plt.title("Train accuracy")
#         plt.xlabel("Step (Epoch)")
#         plt.ylabel("Accuracy")

#         # Test Accuracy
#         plt.subplot(2, 2, 4)
#         sns.lineplot(data=model_data['test_acc'], x='Step', y='Value', 
#                      color=color,
#                      alpha=ALPHA)
#         sns.lineplot(data=model_data['test_acc'], x='Step', y=model_data['test_acc']['Value'].ewm(alpha=SMOOTHING, 
#                                                                                                       adjust=False).mean(), 
#                      label=model_name, 
#                      color=color)
#         plt.title("Test accuracy")
#         plt.xlabel("Step (Epoch)")
#         plt.ylabel("Accuracy")

#     plt.tight_layout()
#     plt.legend(loc='best',
#                ncols=ncols,
#               fontsize=10)
#     plt.show()


# In[35]:


# def multi_plotter3(data_dict: dict,
#                    ncols:int = 3,
#                    fontsize:int =8,
#                    SMOOTHING: float = 0.1):
#     """
#     Generates a 2x2 figure with subplots for train loss, test loss,
#     train accuracy, and test accuracy for multiple models, with colors
#     specified inline within the data dictionary.

#     Args:
#         data_dict (dict): A dictionary where keys are model names (str)
#                            and values are dictionaries containing the DataFrames
#                            for 'train_loss', 'test_loss', 'train_acc', 'test_acc',
#                            and the color associated with the model name (e.g., 'Model 16': '#D33B90').
#                            Each of the DataFrames MUST have 'Step' and 'Value' columns.
#         SMOOTHING (float): Pandas ewm (Exponentially Weighted Moving) smoothing. The higher the value, the lower
#                             the smoothing is. It adjusts ewm alpha parameter.
#                            Example:
#                            {
#                                'Model 16': {'train_loss': df16_train_loss, 
#                                             'test_loss': df16_test_loss, 
#                                             'train_acc': df16_train_acc, 
#                                             'test_acc': df16_test_acc, 
#                                             'color': '#D33B90'},
                                            
#                                'Model 17': {'train_loss': df17_train_loss, 
#                                             'test_loss': df17_test_loss, 
#                                             'train_acc': df17_train_acc, 
#                                             'test_acc': df17_test_acc, 
#                                             'color': '#EDAF3D'},
                                            
#                                'Model 18': {'train_loss': df18_train_loss, 
#                                             'test_loss': df18_test_loss, 
#                                             'train_acc': df18_train_acc, 
#                                             'test_acc': df18_test_acc, 
#                                             'color': '#883ADE'}
#                            }
#     """
#     plt.figure(figsize=(14, 9), dpi=100)

#     for model_name, model_data in data_dict.items():
#         color = model_data.get('color')
        
#         ALPHA=0.2
#         # SMOOTHING=0.1
#         # Train Loss
#         plt.subplot(4, 1, 1)
#         sns.lineplot(data=model_data['train_loss'], x='Step', y='Value', 
#                      color=color,
#                      alpha=ALPHA)
#         sns.lineplot(data=model_data['train_loss'], x='Step', y=model_data['train_loss']['Value'].ewm(alpha=SMOOTHING, 
#                                                                                                       adjust=False).mean(), 
#                      label=model_name, 
#                      color=color)
#         plt.title("Train loss")
#         plt.xlabel("Step (Epoch)")
#         plt.ylabel("Loss")
#         plt.legend(loc='best', ncols=ncols)

#         # Test Loss
#         plt.subplot(4, 1, 2)
#         sns.lineplot(data=model_data['test_loss'], x='Step', y='Value', 
#                      color=color,
#                      alpha=ALPHA)
#         sns.lineplot(data=model_data['test_loss'], x='Step', y=model_data['test_loss']['Value'].ewm(alpha=SMOOTHING, 
#                                                                                                       adjust=False).mean(), 
#                      label=model_name, 
#                      color=color)
#         plt.title("Test loss")
#         plt.xlabel("Step (Epoch)")
#         plt.ylabel("Loss")
#         plt.legend(loc='best', ncols=ncols)

#         # Train Accuracy
#         plt.subplot(4, 1, 3)
#         sns.lineplot(data=model_data['train_acc'], x='Step', y='Value', 
#                      color=color,
#                      alpha=ALPHA)
#         sns.lineplot(data=model_data['train_acc'], x='Step', y=model_data['train_acc']['Value'].ewm(alpha=SMOOTHING, 
#                                                                                                       adjust=False).mean(), 
#                      label=model_name, 
#                      color=color)
#         plt.title("Train accuracy")
#         plt.xlabel("Step (Epoch)")
#         plt.ylabel("Accuracy")
#         plt.legend(loc='best', ncols=ncols)

#         # Test Accuracy
#         plt.subplot(4, 1, 4)
#         sns.lineplot(data=model_data['test_acc'], x='Step', y='Value', 
#                      color=color,
#                      alpha=ALPHA)
#         sns.lineplot(data=model_data['test_acc'], x='Step', y=model_data['test_acc']['Value'].ewm(alpha=SMOOTHING, 
#                                                                                                       adjust=False).mean(), 
#                      label=model_name, 
#                      color=color)
#         plt.title("Test accuracy")
#         plt.xlabel("Step (Epoch)")
#         plt.ylabel("Accuracy")
#         plt.legend(loc='best', ncols=ncols,
#                   fontsize=10)

#     plt.tight_layout()
#     plt.show()


# In[ ]:




