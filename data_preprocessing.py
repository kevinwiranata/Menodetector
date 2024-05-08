import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def combine_datasets():
    # getting the dataframes in a list dfs, and checking the overlapping columns.
    # Create the new dataset with only one column 'race'
    new_dataset = pd.DataFrame(columns=['id', 'race'])
    # Load race columns from other datasets
    dataset1 = pd.read_csv('data/5.tsv', delimiter='\t', low_memory=False)
    dataset2 = pd.read_csv('data/6.tsv', delimiter='\t', low_memory=False)
    dataset3 = pd.read_csv('data/7.tsv', delimiter='\t', low_memory=False)
    dataset4 = pd.read_csv('data/8.tsv', delimiter='\t', low_memory=False)
    dataset5 = pd.read_csv('data/9.tsv', delimiter='\t', low_memory=False)
    dataset6 = pd.read_csv('data/10.tsv', delimiter='\t', low_memory=False)

    race_columns = [dataset1[['SWANID', 'RACE']], dataset2[['SWANID', 'RACE']], dataset3[['SWANID', 'RACE']], dataset4[['SWANID', 'RACE']], dataset5[['SWANID', 'RACE']], dataset6[['SWANID', 'RACE']]]
    stacked_race = pd.concat(race_columns)
    new_dataset[['id', 'race']] = stacked_race
    new_dataset.reset_index(drop=True, inplace=True)
    return new_dataset

def frequency_of_participation(df):
    # Assuming merged_df_cleaned contains a column named 'SWANID' that identifies rows
    # Count how many rows each SWANID has
    swanid_counts = df['id'].value_counts()
    # Count the frequency of each value in the column
    value_counts = swanid_counts.value_counts()
    value_counts = value_counts.sort_index()
    plt.bar(value_counts.index, value_counts.values)
    plt.xlabel('Years of Participation')
    plt.ylabel('Number of participants')
    plt.title('Count of years of participation for each participants')

    for i, v in enumerate(value_counts.values):
        plt.text(i + 1, v + 0.1, str(v), ha='center')
    plt.show()


def show_income(df):
    value_meanings = {
        ' ': 'Null',
        '-9': 'Null',
        '-8': 'Null',
        '-7': 'Null',
        '1': 'LESS THAN $19,999',
        '2': '$20,000 TO $49,999',
        '3': '$50,000 TO $99,999',
        '4': '$100,000 OR MORE'
    }

    desired_order = ['LESS THAN $19,999', '$20,000 TO $49,999', '$50,000 TO $99,999', '$100,000 OR MORE', 'Null']


    # Map the data to their meanings
    data_labels = [value_meanings[val] for val in df["INCOME1"]]

    # Plot countplot with custom x-axis labels
    sns.countplot(x=data_labels, order=desired_order)

    plt.xlabel('Income Range')
    plt.ylabel('Count')
    plt.title('Count of Income Distribution')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

def show_race(unique_ids):
    value_meanings_race = {
        5: 'Hispanic',
        1: 'Black/African American',
        2: 'Chinese/Chinese American',
        3: 'Japanese/Japanese American',
        4: 'Caucasian/White Non-Hispanic'
    }
    data_labels_race = [value_meanings_race[val] for val in unique_ids['race']]
    counts = Counter(data_labels_race)
    labels = counts.keys()
    sizes = counts.values()

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Count of Values')
    plt.axis('equal')

    plt.show()

def demographic_research():
    df = combine_datasets()
    frequency_of_participation(df)
    unique_ids = df.drop_duplicates(subset=['id'])
    show_race(unique_ids)
