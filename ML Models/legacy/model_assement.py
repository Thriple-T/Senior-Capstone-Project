import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from pipeline import load_and_clean_data, engineer_features

def evaluate_models(file1, file2, model_path):
    print("Loading data and engineering features for evaluation...")
    
    try:
        df, targets = load_and_clean_data(file1, file2)
        X, _ = engineer_features(df)
    except Exception as e:
        print(f"Error processing data: {e}")
        return
    
    _, X_test, _, indices_test = train_test_split(
        X, df.index, test_size=0.2, random_state=42
    )
    
    print(f"\nLoading trained models from {model_path}...")
    try:
        models = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Could not find '{model_path}'. Please run pipeline.py first.")
        return

    print("\n" + "="*50)
    print(" SayarDesk Model Accuracy Report ")
    print("="*50)
    
    for target in targets:
        y_test = df.loc[indices_test, target]
        model = models[target]
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Calculate standard evaluation metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Calculate Human Agreement Metric: 
        # In IELTS, two human examiners are considered in agreement if their scores are within 0.5 bands of each other.
        within_half_band = np.sum(np.abs(predictions - y_test) <= 0.5) / len(y_test) * 100
        
        # Display results
        print(f"\nCriterion: {target.replace('_', ' ')}")
        print("-" * 40)
        print(f"  RMSE (Root Mean Squared Error) : {rmse:.3f} Bands")
        print(f"  MAE  (Mean Absolute Error)     : {mae:.3f} Bands")
        print(f"  R²   (Explained Variance)      : {r2:.3f}")
        print(f"  Human Agreement (≤ 0.5 diff)   : {within_half_band:.1f}%")

if __name__ == "__main__":
    # Define file paths
    csv_file1 = 'D:\Projects\Senior-Capstone-Project\Datasets\Scraping Data\combined_ielts_essays_fixed.csv'
    csv_file2 = 'D:\Projects\Senior-Capstone-Project\Datasets\AI Generated Dataset\ielts_gemini_fixed.csv'
    saved_model_path = 'D:\Projects\Senior-Capstone-Project\ML Models\sayardesk_ielts_models.joblib'
    
    evaluate_models(csv_file1, csv_file2, saved_model_path)