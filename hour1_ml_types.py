import numpy as np
import pandas as pd

print("="* 50)
print("Machine Learning - Three Types")
print("="* 50)

# Supervised Learning Classification
classification_examples = [
    "Email spam detection       spam or not spam",
    "Fake news detection        Real or Fake",
    "PSL match prediction       Win or Lose",
    "Disease Diagnosis          Sick or Healthy",
    "Sentiment Analysis         Positive or negative" 
]

#supervised learning regression
regression_examples = [
    "House price prediction            RS.5,000,000",
    "Batman runs prediction            67 runs",
    "Temperature prediction            44 Degrees",
    "IELTS score prediction            7.5 band",
    "Sales forecasting                 RS.200,000",
]

#unsupervised learning
unsupervised_examples =[
    "Customer segmentation          Group A,B,C",
    "Anamoly detection              Normal or outlier",
    "News topic clustering          Politics, sports, tech"
]

print("\n supervised - Classification (output is a category ):")
for ex in classification_examples :
    print(f"   {ex}")

print("\n supervised - regression (output is a number):")
for ex in regression_examples:
    print(f"    {ex}")

print("\n unsupervised (no correct answers given):")
for ex in unsupervised_examples:
    print(f"     {ex}")

#The ML Workflow
print("\n" + "=" * 50)
print("ML WORKFLOW — 7 Steps")
print("=" * 50)
workflow = [
    ("1", "Get Data",           "CSV, database, API"),
    ("2", "Clean Data",         "Missing values, fix types"),
    ("3", "Split Train/Test",   "80% train, 20% test"),
    ("4", "Train Model",        "Model learns from training data"),
    ("5", "Evaluate",           "Test accuracy, precision, recall"),
    ("6", "Improve",            "Tune, retry, compare models"),
    ("7", "Deploy",             "Streamlit app with live URL"),
]
for num, name, desc in workflow:
    print(f"  Step {num}: {name:20} — {desc}")

print("\nPSL project completed Steps 1 and 2.")
print("August completes Steps 3 through 7.")     