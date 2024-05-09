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


import torch
from captum.attr import IntegratedGradients
import numpy as np
import matplotlib.pyplot as plt

def setup_device():
    """ Set up GPU device if available. """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def visualize_all_symptoms_attributions_gpu(model, input_data, feature_names, output_size):
    device = setup_device()
    model.to(device)
    input_data = input_data.to(device)

    model.eval()
    ig = IntegratedGradients(model)
    
    all_attributions = []
    for symptom_index in range(output_size):
        print(f"Computing attributions for symptom {symptom_index + 1}")
        attributions, _ = ig.attribute(input_data, target=symptom_index, return_convergence_delta=True)
        all_attributions.append(attributions.detach().cpu().numpy())  # Move data back to CPU for visualization
    
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

# Example usage:
# model and test_input should be prepared beforehand
# visualize_all_symptoms_attributions_gpu(model, test_input, feature_names, output_size)
