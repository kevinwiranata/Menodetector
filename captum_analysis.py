import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients

def visualize_attributions(model, input_data, target_index=0):
    """
    Visualizes the attributions of the input features towards the model's predictions using Captum's IntegratedGradients.

    Args:
    model (torch.nn.Module): The trained model.
    input_data (torch.Tensor): The input tensor to the model.
    target_index (int): The index of the output for which to compute attributions.
    """
    # Ensure model is in evaluation mode
    model.eval()

    # Initialize IntegratedGradients with the model
    ig = IntegratedGradients(model)

    # Compute the attributions using IntegratedGradients
    attributions, delta = ig.attribute(input_data, target=target_index, return_convergence_delta=True)
    attributions = attributions.detach().numpy()

    # Print attributions and convergence delta
    print("Attributions:\n", attributions)
    print("Convergence Delta:", delta.item())

    # Visualize the attributions as a heatmap
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(attributions[0], annot=True, cmap='coolwarm', fmt=".2f")
    ax.set_title('Feature Importance by Timestep')
    ax.set_xlabel('Features')
    ax.set_ylabel('Timestep')
    plt.show()


def compute_attributions_for_dataset(model, dataloader, device):
    """
    Computes attributions for the entire dataset using Captum's IntegratedGradients in a batch-wise manner.

    Args:
    model (torch.nn.Module): The trained model.
    dataloader (torch.utils.data.DataLoader): DataLoader containing the dataset.
    device (torch.device): The device (CPU or GPU) the model is running on.
    """
    model.eval()
    ig = IntegratedGradients(model)

    # Store attributions
    all_attributions = []

    for inputs, _ in dataloader:
        inputs = inputs.to(device)
        attributions = ig.attribute(inputs, target=0)  # Compute attributions for target index 0
        all_attributions.append(attributions.cpu().detach().numpy())

    # Combine all attributions
    all_attributions = np.concatenate(all_attributions, axis=0)

    return all_attributions

def plot_average_attributions(all_attributions):
    """
    Plots the average attributions across all samples as a heatmap.

    Args:
    all_attributions (np.array): A numpy array containing attributions for each sample in the dataset.
                                 The expected shape is (num_samples, sequence_length, num_features).
    """
    # Compute the mean attributions across all samples
    mean_attributions = np.mean(all_attributions, axis=0)

    # Plotting the heatmap of average attributions
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(mean_attributions, annot=True, cmap='coolwarm', fmt=".2f")
    ax.set_title('Average Feature Importance by Timestep')
    ax.set_xlabel('Features')
    ax.set_ylabel('Timestep')
    plt.show()
