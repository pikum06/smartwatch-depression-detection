# This script implements the depression detection pipeline using ensemble models. It loads and cleans the data, generates visualizations, trains two ensemble models, evaluates their performance, and saves a comparison plot of their accuracies.
 
# import libraries

import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_cleaning import load_and_clean_data
from graphs import generate_visualizations

# function to run the depression detection pipeline

def detection_pipeline():

    # Data Loading & Graph Generation

    csv_path = "../data/target_dataset.csv"
    clean_df = pd.read_csv(csv_path)
    
    # Extract Features and Target
    
    X = clean_df.drop(columns=["Depression"])
    y = clean_df["Depression"]

    # Model Training & Evaluation 
    # Spliting the data in ratio of 70:30 Train-Test Split

    iterations = 10
    acc_m1, acc_m2 = [], []

    for seed in range(iterations):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state= 42 + seed
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Ensemble Model 1: KNN, Logistic Regression, SVM

        knn = KNeighborsClassifier(n_neighbors=5)
        lr = LogisticRegression(max_iter=1000)
        svm = SVC(random_state=42)
        model1 = VotingClassifier(
            estimators=[("knn", knn), ("lr", lr), ("svm", svm)], voting="hard"
        )
        model1.fit(X_train_scaled, y_train)
        pred1 = model1.predict(X_test_scaled)
        acc_m1.append(accuracy_score(y_test, pred1))

        # Ensemble Model 2: Decision Tree, Naïve Bayes, SVM

        dt = DecisionTreeClassifier(random_state=42)
        nb = GaussianNB()
        model2 = VotingClassifier(
            estimators=[("dt", dt), ("nb", nb), ("svm", svm)], voting="hard"
        )
        model2.fit(X_train_scaled, y_train)
        pred2 = model2.predict(X_test_scaled)
        acc_m2.append(accuracy_score(y_test, pred2))

    mean_acc1 = np.mean(acc_m1) * 100
    mean_acc2 = np.mean(acc_m2) * 100

    print("\nEnsemble Model Results")
    print(
        f"Ensemble Model 1 (KNN, LR, SVM) Average Accuracy: {mean_acc1:.3f}%"
    )
    print(
        f"Ensemble Model 2 (DT, NB, SVM) Average Accuracy: {mean_acc2:.3f}%"
    )

# Saving Trained Models 

    models_dir = "../models"
    os.makedirs(models_dir, exist_ok=True)

    # Fit final scaler and models on dataset

    scaler_final = StandardScaler()
    X_scaled_final = scaler_final.fit_transform(X)

    model1_final = VotingClassifier(
        estimators=[
            ("knn", KNeighborsClassifier(n_neighbors=5)),
            ("lr", LogisticRegression(max_iter=1000)),
            ("svm", SVC(random_state=42)),
        ],
        voting="hard",
    )
    model1_final.fit(X_scaled_final, y)

    model2_final = VotingClassifier(
        estimators=[
            ("dt", DecisionTreeClassifier(random_state=42)),
            ("nb", GaussianNB()),
            ("svm", SVC(random_state=42)),
        ],
        voting="hard",
    )
    model2_final.fit(X_scaled_final, y)

    # Export serialized artifacts

    joblib.dump(scaler_final, os.path.join(models_dir, "scaler.joblib"))
    joblib.dump(
        model1_final, os.path.join(models_dir, "ensemble_model_1.joblib")
    )
    joblib.dump(
        model2_final, os.path.join(models_dir, "ensemble_model_2.joblib")
    )

    print(f"\nModels successfully saved to '{os.path.abspath(models_dir)}'")

if __name__ == "__main__":
    detection_pipeline()