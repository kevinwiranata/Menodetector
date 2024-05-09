import argparse
import torch
import torch.nn as nn
from load_data import load_data
from data_preprocessing import demographic_research
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader, TensorDataset
from captum_analysis import (
    visualize_attributions_bar_plot,
    visualize_all_symptoms_attributions,
    visualize_all_symptoms_attributions_parallel,
    visualize_all_symptoms_attributions_gpu,
)

SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

n_features = 48  # Number of lifestyle features
seq_length = 6  # number of years


def get_args():
    parser = argparse.ArgumentParser(description="Menopause training loop")
    parser.add_argument(
        "-g", "--grid_search", action="store_true", default=False, help="Whether to perform grid search"
    )
    parser.add_argument(
        "-op", "--use_optimal_params", action="store_true", default=False, help="Whether to use optimal params"
    )
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("-hl", "--hidden_layer_size", type=int, default=100, help="Hidden layer size for LSTM")
    parser.add_argument("-s", "--silent", action="store_true", default=False, help="Whether to suppress output")
    args = parser.parse_args()
    return args


class MenopauseSymptomsPredictor(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(MenopauseSymptomsPredictor, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions


def train(model, train_loader, val_loader, epochs, optimizer, loss_function):
    model = model.to(DEVICE)
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = loss_function(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                predictions = model(X_batch)
                val_losses.append(loss_function(predictions, y_batch).item())

    avg_train_loss = sum(train_losses) / len(train_losses)
    avg_val_loss = sum(val_losses) / len(val_losses)
    return avg_train_loss, avg_val_loss


def grid_search(X_train, y_train, X_val, y_val, output_size, param_grid):
    best_loss = float("inf")
    best_params = {}
    for params in tqdm(list(ParameterGrid(param_grid))):
        model = MenopauseSymptomsPredictor(n_features, params["hidden_layer_size"], output_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
        loss_function = nn.MSELoss()

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=params["batch_size"], shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=params["batch_size"], shuffle=False)

        avg_train_loss, _ = train(model, train_loader, val_loader, params["epochs"], optimizer, loss_function)

        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            best_params = params

        print(f"Tested {params}, Train Loss: {avg_train_loss}")

    print(f"Best Parameters: {best_params}, Best Train Loss: {best_loss}")
    return best_params


def main():
    args = get_args()
    if not args.silent:
        demographic_research()
    (X_train, y_train), (X_test, y_test), xticknames, symptomnames = load_data(args)
    X_train, y_train = X_train.to(torch.float32), y_train.to(torch.float32)
    X_test, y_test = X_test.to(torch.float32), y_test.to(torch.float32)

    output_size = y_train.shape[1]

    if args.grid_search:
        param_grid = {
            "epochs": [10, 12, 14, 16, 18, 20],
            "learning_rate": [0.01, 0.005, 0.001, 0.0001],
            "batch_size": [16, 32, 64],
            "hidden_layer_size": [50, 100, 200, 300],
        }
        best_params = grid_search(X_train, y_train, X_test, y_test, output_size, param_grid)
        train_epoch = best_params["epochs"]
        train_batch_size = best_params["batch_size"]
        train_lr = best_params["learning_rate"]
        train_hidden_layer_size = best_params["hidden_layer_size"]
    elif args.use_optimal_params:
        train_epoch = 16
        train_batch_size = 64
        train_lr = 0.001
        train_hidden_layer_size = 100
    else:
        train_epoch = args.epochs
        train_batch_size = args.batch_size
        train_lr = args.learning_rate
        train_hidden_layer_size = args.hidden_layer_size

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=train_batch_size, shuffle=False)

    model = MenopauseSymptomsPredictor(n_features, train_hidden_layer_size, output_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_lr)
    loss_function = nn.MSELoss()

    avg_train_loss, avg_val_loss = train(model, train_loader, val_loader, train_epoch, optimizer, loss_function)
    print(f"Final Training Loss: {avg_train_loss}, Final Validation Loss: {avg_val_loss}")
    # Run Captum Analysis here and other plots
    model.to(DEVICE).float()

    # Prepare data for visualization
    ### TEST_INPUT SHOULD BE THE TRAINING DATASET AS A TENSOR, CONVERTED TO FLOAT32 AND MOVED TO DEVICE ###
    ### MODIFY THE LINE BELOW ONLY ###
    # test_input = lifestyle_tensor.to(torch.float32).to(DEVICE)  # Convert to float32 and move to device

    ######### DO NOT CHANGE THE CODE BELOW #########
    feature_names = [f"Feature {i+1}" for i in range(n_features)]  # Generate feature names
    # visualize_all_symptoms_attributions_parallel(model, test_input, feature_names, output_size)
    # visualize_all_symptoms_attributions_gpu(model, test_input, feature_names, xticknames, output_size)
    # for symptom_index in range(output_size):
    #     print(f"Visualizing attributions for symptom {symptom_index + 1}")
    #     visualize_attributions_bar_plot(model, test_input, feature_names, xticknames, target_index=symptom_index)


if __name__ == "__main__":
    main()
