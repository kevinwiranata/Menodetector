import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR


def random_forest_baseline(features_time_avg, targets):
    targets = targets.view(-1) if targets.dim() > 1 else targets

    # Prepare K-Fold cross-validation
    num_folds = 10
    kfold = KFold(n_splits=num_folds, shuffle=True)

    # Cross-validation process for Random Forest
    rf_results = []
    for train_ids, test_ids in kfold.split(features_time_avg):
        # Split data
        train_features = features_time_avg[train_ids]
        train_targets = targets[train_ids]
        test_features = features_time_avg[test_ids]
        test_targets = targets[test_ids]

        # Initialize and train the Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100)
        rf_model.fit(train_features, train_targets)

        # Validation
        predictions = rf_model.predict(test_features)
        val_loss = mean_squared_error(test_targets, predictions)
        rf_results.append(val_loss)

        print(f"Random Forest - Fold Test Loss: {val_loss}")

    # Calculate and print average loss across folds
    rf_average_loss = np.mean(rf_results)
    print(f"Random Forest - Average Loss across folds: {rf_average_loss}")


def SVR_baseline(features_time_avg, targets):
    targets = targets.view(-1) if targets.dim() > 1 else targets
    # Cross-validation process for SVR
    svr_results = []
    num_folds = 10
    kfold = KFold(n_splits=num_folds, shuffle=True)
    for train_ids, test_ids in kfold.split(features_time_avg):
        # Split data
        train_features = features_time_avg[train_ids]
        train_targets = targets[train_ids]
        test_features = features_time_avg[test_ids]
        test_targets = targets[test_ids]

        # Initialize and train the SVR model
        svr_model = SVR(kernel="rbf", C=1.0, epsilon=0.1)
        svr_model.fit(train_features, train_targets)

        # Validation
        predictions = svr_model.predict(test_features)
        val_loss = mean_squared_error(test_targets, predictions)
        svr_results.append(val_loss)

        print(f"SVR - Fold Test Loss: {val_loss}")

    # Calculate and print average loss across folds
    svr_average_loss = np.mean(svr_results)
    print(f"SVR - Average Loss across folds: {svr_average_loss}")
