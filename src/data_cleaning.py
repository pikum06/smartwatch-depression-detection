# Data cleaning script for the Confluence 2021 research project.
# This script loads the raw dataset, removes irrelevant features, and encodes categorical columns for further analysis.

# import libraries

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# function to load and clean the dataset

def load_and_clean_data(file_path="../data/cleaned_data.xlsx"):
    "Loads raw dataset, removes irrelevant features, and encodes categorical columns."
    df = pd.read_excel(file_path)

    # Removing irrelevant or low-variance features per Section III-C & III-D
    cols_to_remove = [
        "How many days were you hospitalized for your mental illness",
        "Region",
        "Household Income",
        "Device Type",
        "I am on section 8 housing",
        "I receive food stamps",
    ]

    df_cleaned = df.drop(
        columns=[col for col in cols_to_remove if col in df.columns]
    )
    df_cleaned = df_cleaned.fillna(0)

    # Categorical label encoding
    categorical_cols = ["Education", "Age", "Gender"]
    for col in categorical_cols:
        if col in df_cleaned.columns:
            le = LabelEncoder()
            df_cleaned[col] = le.fit_transform(df_cleaned[col].astype(str))

    return df, df_cleaned


if __name__ == "__main__":
    raw_df, clean_df = load_and_clean_data()
    clean_df.to_csv("../data/target_dataset.csv", index=False)
    print(f"Data cleaning complete. Output shape: {clean_df.shape}")