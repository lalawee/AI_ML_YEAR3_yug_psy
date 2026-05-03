import pandas as pd
import joblib
import argparse
from sklearn.metrics import classification_report
from preprocessing import preprocess_data

def visualize_model_performance(model_path, test_data_path):
    # Load model and data
    model = joblib.load(model_path)
    data = pd.read_csv(test_data_path)

    # Preprocess data using the shared module
    X, y = preprocess_data(data)

    # Predict and evaluate
    y_pred = model.predict(X)
    print("Test set report:\n", classification_report(y, y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model performance on test data.")
    parser.add_argument('--model', type=str, help="Path to the trained model.")
    parser.add_argument('--data', type=str, help="Path to the test dataset.")
    
    args = parser.parse_args()
    visualize_model_performance(args.model, args.data)
