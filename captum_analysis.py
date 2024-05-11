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

# def visualize_all_symptoms_attributions_gpu(model, input_data, feature_names, output_size):
#     device = setup_device()
#     model.to(device)
#     input_data = input_data.to(device)

#     model.eval()
#     ig = IntegratedGradients(model)

#     all_attributions = []
#     for symptom_index in range(output_size):
#         print(f"Computing attributions for symptom {symptom_index + 1}")
#         attributions, _ = ig.attribute(input_data, target=symptom_index, return_convergence_delta=True)
#         all_attributions.append(attributions.detach().cpu().numpy())  # Move data back to CPU for visualization

#     mean_attributions = np.mean(np.array(all_attributions), axis=0)
#     if mean_attributions.ndim > 1:
#         mean_attributions = np.mean(mean_attributions, axis=tuple(range(1, mean_attributions.ndim)))

#     plt.figure(figsize=(10, 5))
#     plt.bar(feature_names, mean_attributions, color='blue')
#     plt.xticks(rotation=90)
#     plt.title("Mean Attributions Across All Symptoms")
#     plt.xlabel("Features")
#     plt.ylabel("Attribution")
#     plt.savefig('captum_attributes_all.png')  # Save the figure as a PNG file
#     plt.show()

# Example usage:
# model and test_input should be prepared beforehand
# visualize_all_symptoms_attributions_gpu(model, test_input, feature_names, output_size)

def visualize_all_symptoms_attributions_gpu(model, input_data, feature_names, xticknames, output_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    input_data = input_data.to(device)

    ig = IntegratedGradients(model)
    all_attributions = []

    model.train()  # Enable train mode to compute gradients
    torch.set_grad_enabled(True)

    for symptom_index in range(output_size):
        print(f"Computing attributions for symptom {symptom_index + 1}")
        attributions, _ = ig.attribute(input_data, target=symptom_index, return_convergence_delta=True)
        all_attributions.append(attributions.detach().cpu().numpy())

    model.eval()
    torch.set_grad_enabled(False)

    # Assuming we need to average across all dimensions except the last one (features)
    mean_attributions = np.mean(np.array(all_attributions), axis=(0, 1, 2))
    print("Shape of mean attributions:", mean_attributions.shape)

    assert len(feature_names) == mean_attributions.shape[0], "Mismatch between number of features and attributions"

    plt.figure(figsize=(12, 8))
    plt.bar(feature_names, mean_attributions, color='blue')
    plt.xticks(feature_names, xticknames, rotation=90)
    plt.title("Mean Attributions Across All Symptoms")
    plt.xlabel("Features")
    plt.ylabel("Attribution")
    plt.savefig('captum_attributes_all.png')
    plt.show()

import os
import shutil

def visualize_symptoms_attributions(model, input_data, feature_names, xticknames, symptom_names):
    """
    Computes and visualizes the attributions of the input features towards the model's predictions for multiple symptoms,
    and saves the plots in a zip file.

    Args:
    model (torch.nn.Module): The trained model.
    input_data (torch.Tensor): The input tensor to the model.
    feature_names (list of str): Names of each feature.
    xticknames (list of str): Names of each feature for x-axis labeling.
    symptom_names (list of str): Names of each symptom.
    """
    # Set up directory for storing images
    folder_name = 'captum'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Ensure model is in evaluation mode and use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    input_data = input_data.to(device)

    # Initialize IntegratedGradients with the model
    ig = IntegratedGradients(model)
    model.train()  # Enable train mode to compute gradients
    torch.set_grad_enabled(True)

    # Process each symptom
    for index, symptom_name in enumerate(symptom_names):
        attributions, delta = ig.attribute(input_data, target=index, return_convergence_delta=True)
        average_attributions = np.mean(attributions.detach().cpu().numpy(), axis=(0, 1))  # Averaging across batch and time steps

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.bar(feature_names, average_attributions)
        plt.xlabel('Features')
        plt.ylabel('Average Attribution')
        plt.title(f'Average Feature Attributions for {symptom_name}')
        plt.xticks(range(len(feature_names)), xticknames, rotation=90)
        file_path = os.path.join(folder_name, f'captum_attributes_{symptom_name}.png')
        plt.savefig(file_path)
        plt.close()  # Close the figure to free memory

        print(f"Attributions for {symptom_name}:\n", average_attributions)
        print(f"Convergence Delta for {symptom_name}:", delta)

    model.eval()
    torch.set_grad_enabled(False)

    # Zip the folder
    shutil.make_archive(folder_name, 'zip', folder_name)

    # Optionally, remove the folder after zipping if desired
    # shutil.rmtree(folder_name)

