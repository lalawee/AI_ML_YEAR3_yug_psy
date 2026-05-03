import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from preprocessing import preprocess_data

def eda(data_path, output_folder):
    # Load data
    data = pd.read_csv(data_path)

    # Preprocess using the shared module; recombine X and y for full-frame analysis
    X, y = preprocess_data(data)
    data = X.copy()
    if y is not None:
        data['Survived'] = y.values

    # Descriptive statistics
    print(data.describe())

    # Missing values
    print("Missing Values:\n", data.isnull().sum())

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.savefig(f"{output_folder}/correlation_heatmap.png")
    plt.close()

    # Distribution plots
    data.hist(figsize=(12, 10), bins=30)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/distribution.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform EDA on a dataset.")
    parser.add_argument('--data', type=str, help="Path to the dataset.")
    parser.add_argument('--output', type=str, help="Folder to save the EDA outputs.")
    
    args = parser.parse_args()
    eda(args.data, args.output)
