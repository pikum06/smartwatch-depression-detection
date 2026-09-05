# Data visualization script for the Confluence 2021 research project.
# This script generates and saves exploratory data figures to the specified output folder.


# import libraries
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# function to generate data figures

def generate_visualizations(df, output_dir="../figures"):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Distribution of people suffering from depression

    plt.figure(figsize=(6, 5))
    sns.countplot(
        x="Depression", 
        hue = "Depression", 
        data=df, 
        palette=["#1f77b4", "#ff7f0e"], 
        legend=False
        )

    plt.title("Distribution of people suffering from depression", fontsize=12)
    plt.xlabel("Depression", fontsize=10)
    plt.ylabel("count", fontsize=10)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "depression_distribution.png"), dpi=300
    )
    plt.close()

    # Distribution of people suffering from anxiety
    plt.figure(figsize=(6, 5))
    sns.countplot(
        x="Anxiety", 
        hue="Anxiety", 
        data=df, 
        palette=["#1f77b4", "#ff7f0e"], 
        legend=False
    )
    plt.title("Distribution of people suffering from anxiety", fontsize=12)
    plt.xlabel("Anxiety", fontsize=10)
    plt.ylabel("count", fontsize=10)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "anxiety_distribution.png"), dpi=300
    )
    plt.close()

    # Pie chart showing symptoms distribution
    symptoms = [
        "Anxiety",
        "Depression",
        "Lack of concentration",
        "Obsessive thinking",
        "Mood swings",
    ]
    symptom_counts = df[symptoms].sum()
    colors = ["#ff7f0e", "#2ca02c", "#1f77b4", "#d62728", "#9467bd"]
    explode = (0.08, 0.04, 0.04, 0, 0)

    plt.figure(figsize=(12, 10))
    plt.pie(
        symptom_counts,
        labels=symptom_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        explode=explode,
        shadow=False,
        colors=colors,
    )
    plt.title("People suffering from different mental illness", fontsize=12)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "symptoms_pie_chart.png"), dpi=300
    )
    plt.close()

    # Pie chart comparing unemployed and part-time workers
    unemployed_count = (df["I am unemployed"] == 1).sum()
    employed_count = (
        df["I am currently employed at least part-time"] == 1
    ).sum()
    emp_counts = [employed_count, unemployed_count]
    emp_labels = [
        "I am currently employed \nat least part-time",
        "I am unemployed",
    ]
    emp_explode = (0, 0.08)
    emp_colors = ["#1f77b4", "#ff7f0e"]

    plt.figure(figsize=(10, 8))
    plt.pie(
        emp_counts,
        labels=emp_labels,
        autopct="%1.1f%%",
        startangle=90,
        explode=emp_explode,
        shadow=False,
        colors=emp_colors,
    )
    plt.title(
        "Comparison between unemployed people and part-timers", fontsize=12
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "employment_pie_chart.png"), dpi=300
    )
    plt.close()

    print(f"All graphs successfully saved to '{output_dir}'.")


if __name__ == "__main__":
    raw_df = pd.read_excel("../data/cleaned_data.xlsx")
    generate_visualizations(raw_df)