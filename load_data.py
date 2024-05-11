import torch
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
from nltk import download
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from data_preprocessing import show_income

download("punkt")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def draw_null_rate(df, data_class, condition):
    null_rate_per_id = df.drop("SWANID1", axis=1).groupby(df["SWANID1"]).apply(lambda x: (x.isna()).mean(axis=1).mean())
    null_rate_per_id.head()
    plt.hist(null_rate_per_id)
    plt.title(f"Null Rate for Patients in {data_class} dataset {condition} Cleaning")
    plt.xlabel("Null Rate")
    plt.ylabel("Count of Patients")
    plt.grid(axis="x")
    plt.show()

    plt.figure(figsize=(10, 10))
    nan_percentage = df[df.columns].isna().mean()
    print(type(nan_percentage))
    nan_percentage.sort_values().plot(kind="barh", color="skyblue")
    plt.title(f"Missing Rate In Each Column in {data_class} dataset {condition} Cleaning")
    plt.xlabel("feature")
    plt.ylabel("missing rate")
    plt.grid(axis="x")
    plt.xticks(fontsize="xx-small")
    plt.show()


def clean_invalid_values(df, columns, data_class, args):
    # Calculate the null rate across features for each unique ID
    if not args.silent:
        draw_null_rate(df, data_class, "before")
    for column in columns:
        if column in df.columns:
            # Strip leading/trailing spaces and replace values with length <= 2 with NaN
            df[column] = df[column].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Calculate the percentage of NaN values in the specified columns
    nan_percentage = df[columns].isna().mean()

    # Drop columns where more than 50% of the values are NaN
    columns_to_drop = nan_percentage[nan_percentage > 0.4].index.tolist()
    print(columns_to_drop)
    df.drop(columns=columns_to_drop, inplace=True)
    nan_percentage_new = df[df.columns].isna().mean()
    print(type(nan_percentage_new))
    if not args.silent:
        draw_null_rate(df, data_class, "after")
    # nan_percentage_new.plot(kind="bar")
    # plt.title(f"Missing Rate In Each Column for {data_class}")
    # plt.xlabel("feature")
    # plt.ylabel("missing rate")
    # plt.xticks(fontsize="small")
    # plt.show()

    return df


def replace_negatives_in_all_columns(df):
    """
    Attempts to convert all columns in the DataFrame to integer.
    If conversion is successful, replaces negative values with NaN.

    Parameters:
    - df: pandas.DataFrame - The DataFrame to modify.

    Returns:
    - The DataFrame with negative integer values replaced by NaN across all columns.
    """
    for column in df.columns:
        try:
            # Attempt to convert column to integer, allowing NaN for conversion failures
            if column != "SWANID1":
                converted_column = pd.to_numeric(df[column], errors="coerce", downcast="integer")
                # Replace negative values with NaN
                df[column] = np.where(converted_column < 0, np.nan, converted_column)
        except Exception as e:
            print(f"Error converting {column}: {e}")
            # If conversion fails, the column is left unchanged

    return df


def convert_columns_to_numeric(df):
    """
    Attempts to convert each column in the DataFrame to a numeric type if possible.

    Parameters:
    - df: pandas.DataFrame

    Returns:
    - A DataFrame with columns converted to numeric types where possible.
    """
    for column in df.columns:
        if column != "SWANID1":
            df[column] = pd.to_numeric(df[column], errors="ignore")
    return df


def data_imputation(df, data_type):
    for column in df.columns:
        if column not in ["SWANID1", "AGE1"]:
            # If it is a categorical column, replace np.nan with mode of that column
            if column in data_type["categorical"]:
                mode_value = df[column].mode()[0]
                # print("mode is", mode_value)
                df[column].fillna(mode_value, inplace=True)
            # If it is a numerical column, replace np.nan with median of that column
            elif column in data_type["numerical"]:
                median_value = df[column].median()
                # print("median is", median_value)
                df[column].fillna(median_value, inplace=True)
    return df


def data_imputation(df, data_type):
    for column in df.columns:
        if column not in ["SWANID1", "AGE1"]:
            if column in data_type["categorical"]:
                # If it is a categorical column, replace np.nan with mode of that column
                mode_value = df[column].mode()[0]
                df[column] = df[column].fillna(mode_value)
            elif column in data_type["numerical"]:
                # If it is a numerical column, replace np.nan with median of that column
                median_value = df[column].median()
                df[column] = df[column].fillna(median_value)
    return df


def transform_to_tensor(df, mode):
    # Define the target shape parameters
    max_entries_per_group = 6

    # Placeholder list for group tensors
    group_tensors = []
    for name, group in df.groupby("SWANID1"):
        group_sorted = group.sort_values("AGE1")
        group_dropped = group_sorted.drop(["SWANID1", "AGE1"], axis=1)

        if mode == "symptom":
            group_dropped = group_dropped.iloc[-1]
        else:
            # Pad the group if it has less than 6 entries
            num_padding_rows = max_entries_per_group - len(group_dropped)
            if num_padding_rows > 0:
                padding = pd.DataFrame(0, index=range(num_padding_rows), columns=group_dropped.columns)
                group_dropped = pd.concat([group_dropped, padding], ignore_index=True)

        # Convert the group to a tensor and add it to the list
        group_tensor = torch.tensor(group_dropped.values)
        group_tensors.append(group_tensor)

    # Stack all group tensors to form the final tensor
    final_tensor = torch.stack(group_tensors)

    # Reshape to desired format (number of groups, 6, number of columns)
    if mode == "lifestyle":
        final_tensor = final_tensor.reshape(-1, max_entries_per_group, final_tensor.shape[-1])
    else:
        final_tensor = final_tensor.reshape(-1, final_tensor.shape[-1])
    return final_tensor


# preprocess the data
def preprocess(df, data_class, args):

    stemmer = PorterStemmer()
    data_type = {"categorical": [], "numerical": []}

    def is_valid_number(val):
        try:
            num = float(val)
            return num != -1
        except ValueError:
            return False

    def stem_textual_values(value):
        if isinstance(value, str) and value != "-1":
            return stemmer.stem(value)
        return value

    def is_number(value):
        try:
            float(value.strip())
            return True
        except ValueError:
            return False

    for column in df.columns:
        if column not in ["SWANID1", "AGE1"]:
            df[column] = df[column].astype(str)
            df[column] = df[column].str.strip().str.upper()
            # For columns with categorical values
            if df[column].apply(lambda x: not is_number(x) or pd.isna(x) or len(x) == 0).all():
                df[column] = df[column].apply(stem_textual_values).astype("category").cat.codes + 1
                data_type["categorical"].append(column)

            # For columns with numerical values
            else:

                if df[column].apply(is_valid_number).mean() >= 0.9:
                    # Convert the entire column to numeric, forcing non-convertible values to NaN
                    df[column] = pd.to_numeric(df[column], errors="coerce")
                    # Find the max value excluding -1
                    max_val = df[column].astype("float").max()
                    # Normalize values in the range [0,1] excluding -1, directly without applying lambda
                    df[column] = np.where(df[column] != "-1", df[column].astype("float") / max_val, df[column])
                data_type["numerical"].append(column)

    df = replace_negatives_in_all_columns(df)
    df = clean_invalid_values(df, df.columns, data_class, args)
    df = data_imputation(df, data_type)
    df = df.dropna(subset=["AGE1"])
    for column in df.columns:
        if df[column].isna().mean() > 0:
            print(column, "has null")

    return df


def load_data(args, test_size=0.2, random_state=None):
    symptom_df = pd.read_csv("data/merged_symptom_only.tsv", sep="\t")
    life_style_df = pd.read_csv("data/merged_lifestyle_only.tsv", sep="\t")
    if not args.silent:
        show_income(life_style_df)
    symptom = preprocess(symptom_df, "Symptom", args)
    life_style = preprocess(life_style_df, "Lifestyle", args)
    symptom = symptom.drop(["PREGNAN1", "PRGNANT1", "BROKEBO1"], axis=1)

    with open("symptom_cols.txt", "w") as f:
        print(f"Saving {len(symptom.columns)} symptom column names")
        for column in symptom.columns:
            f.write(f"{column}\n")

    with open("lifestyle_cols.txt", "w") as f:
        print(f"Saving {len(life_style.columns)} lifestyle column names")
        for column in life_style.columns:
            f.write(f"{column}\n")

    lifestyle_tensor = transform_to_tensor(life_style, "lifestyle")
    symptom_tensor = transform_to_tensor(symptom, "symptom")

    # Perform train/test split
    indices = list(range(len(lifestyle_tensor)))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state, shuffle=True)

    # Prepare the train and test data
    X_train, X_test = lifestyle_tensor[train_idx], lifestyle_tensor[test_idx]
    y_train, y_test = symptom_tensor[train_idx], symptom_tensor[test_idx]
    print(len(symptom.columns), symptom.columns)
    print(len(X_train[0]))

    return (
        (X_train, y_train),
        (X_test, y_test),
        lifestyle_tensor,
        symptom_tensor,
        life_style_df.columns.tolist()[2:],
        symptom.columns.tolist()[2:],
    )
