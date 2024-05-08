import argparse
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from load_data import load_data
from data_preprocessing import demographic_research
import torch


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
    # Training hyperparameters
    parser.add_argument("-g", "--grid_search", type=bool, default=False, help="Whether to perform grid search or not")
    parser.add_argument(
        "-op", "--use_optimal_params", type=bool, default=False, help="Whether to use optimal params or not"
    )
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs to train the model")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("-hl", "--hidden_layer_size", type=int, default=100, help="Hidden layer size for LSTM")
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
        predictions = self.linear(lstm_out[:, -1, :])  # last time step's output
        return predictions


def tr(model_class, X, y, epochs, batch_size, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_idx, val_idx = train_test_split(range(len(X)), test_size=0.2, shuffle=True)
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
    train_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(X, y), batch_size=batch_size, sampler=train_sampler
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(X, y), batch_size=batch_size, sampler=val_sampler
    )

    model = model_class.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.MSELoss()

    for _ in range(epochs):
        model.train()
        train_losses = []  # List to store train losses for each batch
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = loss_function(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)  # Average loss of the last epoch

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                val_losses.append(loss_function(predictions, y_batch).item())

    avg_val_loss = sum(val_losses) / len(val_losses)

    # Return both average validation loss and average training loss of the last epoch
    return avg_val_loss, avg_train_loss


import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split, ParameterGrid


def grid_search(
    X,
    y,
    output_size,
    param_grid,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid = ParameterGrid(param_grid)
    best_loss = float("inf")
    best_params = {}
    all_records = []

    for params in tqdm(grid):
        model_class = MenopauseSymptomsPredictor(n_features, params["hidden_layer_size"], output_size).to(device)
        print(f"Testing with parameters: {params}")
        avg_val_loss, avg_train_loss = tr(
            model_class,
            X,
            y,
            params["epochs"],
            params["batch_size"],
            params["learning_rate"],
        )
        print(f"Average Validation Loss: {avg_val_loss}")
        print(f"Average Training Loss: {avg_train_loss}")
        all_records.append((params, avg_val_loss, avg_train_loss))
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            best_params = params

    return best_params, best_loss, all_records


def main():

    args = get_args()
    demographic_research()
    lifestyle_tensor, symptom_tensor = load_data()
    output_size = symptom_tensor.shape[1]
    train_epoch, train_batch_size, train_lr, train_hidden_layer_size = None, None, None, None

    # Define the parameter grid
    if args.grid_search:
        param_grid = {
            "epochs": [10, 12, 14, 16, 18, 20],
            "learning_rate": [0.01, 0.005, 0.001, 0.0001],
            "batch_size": [16, 32, 64],
            "hidden_layer_size": [50, 100, 200, 300],
        }
        best_params, best_loss, all_records = grid_search(
            lifestyle_tensor.to(torch.float32), symptom_tensor.to(torch.float32), output_size, param_grid
        )
        train_epoch = best_params["epochs"]
        train_batch_size = best_params["batch_size"]
        train_lr = best_params["learning_rate"]
        train_hidden_layer_size = best_params["hidden_layer_size"]

        print(all_records)
        with open("result.txt", "w") as file:
            file.write(str(all_records))
        print(f"Best Parameters: {best_params}")
        print(f"Lowest Validation Loss: {best_loss}")

    elif args.use_optimal_params:
        train_epoch = 18
        train_batch_size = 16
        train_lr = 0.001
        train_hidden_layer_size = 100
    else:
        train_epoch = args.epochs
        train_batch_size = args.batch_size
        train_lr = args.learning_rate
        train_hidden_layer_size = args.hidden_layer_size
    model = MenopauseSymptomsPredictor(n_features, train_hidden_layer_size, output_size).to(DEVICE)

    # Train model with best params
    tr(
        model,
        lifestyle_tensor.to(torch.float32),
        symptom_tensor.to(torch.float32),
        train_epoch,
        train_batch_size,
        train_lr,
    )

    # Run Captum Analysis here and other plots


if __name__ == "__main__":
    main()
