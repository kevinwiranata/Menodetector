# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from captum.attr import IntegratedGradients

# def visualize_attributions(model, input_data, target_index=0):
#     """
#     Visualizes the attributions of the input features towards the model's predictions using Captum's IntegratedGradients.

#     Args:
#     model (torch.nn.Module): The trained model.
#     input_data (torch.Tensor): The input tensor to the model.
#     target_index (int): The index of the output for which to compute attributions.
#     """
#     # Ensure model is in evaluation mode
#     model.eval()

#     # Initialize IntegratedGradients with the model
#     ig = IntegratedGradients(model)

#     # Compute the attributions using IntegratedGradients
#     attributions, delta = ig.attribute(input_data, target=target_index, return_convergence_delta=True)
#     attributions = attributions.detach().numpy()

#     # Print attributions and convergence delta
#     print("Attributions:\n", attributions)
#     print("Convergence Delta:", delta.item())

#     # Visualize the attributions as a heatmap
#     plt.figure(figsize=(12, 6))
#     ax = sns.heatmap(attributions[0], annot=True, cmap='coolwarm', fmt=".2f")
#     ax.set_title('Feature Importance by Timestep')
#     ax.set_xlabel('Features')
#     ax.set_ylabel('Timestep')
#     plt.show()


# def compute_attributions_for_dataset(model, dataloader, device):
#     """
#     Computes attributions for the entire dataset using Captum's IntegratedGradients in a batch-wise manner.

#     Args:
#     model (torch.nn.Module): The trained model.
#     dataloader (torch.utils.data.DataLoader): DataLoader containing the dataset.
#     device (torch.device): The device (CPU or GPU) the model is running on.
#     """
#     model.eval()
#     ig = IntegratedGradients(model)

#     # Store attributions
#     all_attributions = []

#     for inputs, _ in dataloader:
#         inputs = inputs.to(device)
#         attributions = ig.attribute(inputs, target=0)  # Compute attributions for target index 0
#         all_attributions.append(attributions.cpu().detach().numpy())

#     # Combine all attributions
#     all_attributions = np.concatenate(all_attributions, axis=0)

#     return all_attributions

# def plot_average_attributions(all_attributions):
#     """
#     Plots the average attributions across all samples as a heatmap.

#     Args:
#     all_attributions (np.array): A numpy array containing attributions for each sample in the dataset.
#                                  The expected shape is (num_samples, sequence_length, num_features).
#     """
#     # Compute the mean attributions across all samples
#     mean_attributions = np.mean(all_attributions, axis=0)

#     # Plotting the heatmap of average attributions
#     plt.figure(figsize=(12, 6))
#     ax = sns.heatmap(mean_attributions, annot=True, cmap='coolwarm', fmt=".2f")
#     ax.set_title('Average Feature Importance by Timestep')
#     ax.set_xlabel('Features')
#     ax.set_ylabel('Timestep')
#     plt.show()

import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
import os
from concurrent.futures import ThreadPoolExecutor


def visualize_attributions_bar_plot(model, input_data, feature_names, xticknames, symptomname, target_index=0):
    """
    Visualizes the attributions of the input features towards the model's predictions using Captum's IntegratedGradients.
    Averages attributions across both samples and time steps for each feature and displays them in a bar plot.

    Args:
    model (torch.nn.Module): The trained model.
    input_data (torch.Tensor): The input tensor to the model.
    feature_names (list of str): Names of each feature.
    target_index (int): The index of the output for which to compute attributions.
    """
    # Ensure model is in evaluation mode
    model.eval()

    # Initialize IntegratedGradients with the model
    ig = IntegratedGradients(model)

    # Compute the attributions using IntegratedGradients
    attributions, delta = ig.attribute(input_data, target=target_index, return_convergence_delta=True)
    attributions = attributions.detach().numpy()

    # Average attributions across both samples and time steps
    average_attributions = np.mean(attributions, axis=(0, 1))  # Averaging across batch and time steps

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.bar(feature_names, average_attributions)
    plt.xlabel('Features')
    plt.ylabel('Average Attribution')
    plt.title(f'Average Feature Attributions across Time Steps and Samples for {symptomname}')
    plt.xticks(feature_names, xticknames, rotation=90)  # Rotate feature names for better visibility
    plt.show()

    # Print attributions and convergence delta
    print("Attributions:\n", average_attributions)
    print("Convergence Delta:", delta)

def visualize_all_symptoms_attributions(model, input_data, feature_names, output_size):
    """
    Visualizes the mean attributions across all symptoms for a given model using Captum's IntegratedGradients.

    Args:
        model (torch.nn.Module): The trained model.
        input_data (torch.Tensor): The input tensor to the model.
        feature_names (list of str): Names of each feature.
        output_size (int): The number of symptoms (output classes) to compute attributions for.
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Initialize IntegratedGradients with the model
    ig = IntegratedGradients(model)

    # Calculate attributions for each symptom and store them in a list
    all_attributions = []
    for symptom_index in range(output_size):
        attributions, _ = ig.attribute(input_data, target=symptom_index, return_convergence_delta=True)
        all_attributions.append(attributions.cpu().detach().numpy())

    # Convert list to numpy array and calculate mean across all symptoms
    mean_attributions = np.mean(np.array(all_attributions), axis=0)

    # Flatten the mean attributions in case of extra dimensions
    if mean_attributions.ndim > 1:
        mean_attributions = np.mean(mean_attributions, axis=tuple(range(1, mean_attributions.ndim)))

    # Plotting the mean attributions
    plt.figure(figsize=(10, 5))
    plt.bar(feature_names, mean_attributions, color='blue')
    plt.xticks(rotation=90)
    plt.title("Mean Attributions Across All Symptoms")
    plt.xlabel("Features")
    plt.ylabel("Attribution")
    plt.show()

# Example usage:
# visualize_all_symptoms_attributions(model, test_input, feature_names, output_size)

def compute_attributions(model, input_data, target_index, ig):
    attributions, _ = ig.attribute(input_data, target=target_index, return_convergence_delta=True)
    return attributions.cpu().detach().numpy()

def visualize_all_symptoms_attributions_parallel(model, input_data, feature_names, output_size, num_threads=os.cpu_count()):
    model.eval()
    ig = IntegratedGradients(model)
    
    # Use ThreadPoolExecutor to parallelize attribution computation
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(compute_attributions, model, input_data, idx, ig) for idx in range(output_size)]
        all_attributions = [future.result() for future in futures]
    
    mean_attributions = np.mean(np.array(all_attributions), axis=0)
    if mean_attributions.ndim > 1:
        mean_attributions = np.mean(mean_attributions, axis=tuple(range(1, mean_attributions.ndim)))
    
    plt.figure(figsize=(10, 5))
    plt.bar(feature_names, mean_attributions, color='blue')
    plt.xticks(rotation=90)
    plt.title("Mean Attributions Across All Symptoms")
    plt.xlabel("Features")
    plt.ylabel("Attribution")
    plt.show()