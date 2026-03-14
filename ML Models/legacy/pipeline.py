import pandas as pd
import numpy as np
import re
import joblib
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sentence_transformers import SentenceTransformer

def load_and_clean_data(file1, file2):
    print("Loading and merging datasets...")
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df = pd.concat([df1, df2], ignore_index=True)
    
    targets = ['Task_Achievement', 'Coherence_Cohesion', 'Lexical_Resource', 'Grammar_Range']
    df = df.dropna(subset=['Essay_Text'] + targets)
    
    df['Type'] = df['Type'].astype(str).str.lower()
    
    print(f"Total essays ready for processing: {len(df)}")
    return df, targets

def extract_handcrafted_features(text):
    """
    Extracts traditional linguistic features (Luan & Tsai, 2021).
    """
    # Clean text for word counting
    words = re.findall(r'\b\w+\b', text.lower())
    sentences = re.split(r'[.!?]+', text)
    
    word_count = len(words)
    sentence_count = len([s for s in sentences if len(s.strip()) > 0])
    
    # Prevent division by zero
    if word_count == 0: word_count = 1 
    if sentence_count == 0: sentence_count = 1
        
    # Type-Token Ratio (TTR) - measures vocabulary richness
    unique_words = len(set(words))
    ttr = unique_words / word_count
    
    # Average sentence length - measures syntactic complexity
    avg_sentence_length = word_count / sentence_count
    
    return [word_count, ttr, avg_sentence_length]

def engineer_features(df):
    print("Engineering Handcrafted Linguistic Features...")
    handcrafted = df['Essay_Text'].apply(extract_handcrafted_features)
    hc_features_df = pd.DataFrame(handcrafted.tolist(), columns=['Word_Count', 'TTR', 'Avg_Sentence_Length'])
    
    print("Extracting Deep Learning Embeddings (BERT)...")
    # Using 'all-MiniLM-L6-v2' - A lightweight BERT model perfect for local/low-resource deployment for now...
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Encode the essays into semantic vectors (384 dimensions)
    embeddings = bert_model.encode(df['Essay_Text'].tolist(), show_progress_bar=True)
    embeddings_df = pd.DataFrame(embeddings, columns=[f'BERT_{i}' for i in range(embeddings.shape[1])])
    
    # Combine Handcrafted and BERT features (The "Hybrid Approach" - Faseeh et al., 2024)
    X = pd.concat([hc_features_df, embeddings_df], axis=1)
    
    return X, hc_features_df.columns.tolist()

# Training the LwXGBoost Model
def train_evaluate_save_models(X, df, targets):
    models = {}
    X_train, X_test, indices_train, indices_test = train_test_split(
        X, df.index, test_size=0.2, random_state=42
    )
    
    print("\nTraining LwXGBoost Models for IELTS Criteria...")
    
    for target in targets:
        y_train = df.loc[indices_train, target]
        y_test = df.loc[indices_test, target]
        
        # Configure Lightweight XGBoost Regressor
        # max_depth=3 prevents overfitting on small datasets
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=3, 
            random_state=42,
            objective='reg:squarederror'
        )
        
        xgb_model.fit(X_train, y_train)
        
        predictions = xgb_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print(f"[{target}] Model Trained. Test RMSE: {rmse:.2f} Bands")
        
        models[target] = xgb_model
        
    return models, X_train, X_test

# Explainability (SHAP)
def generate_shap_explanations(models, X_test, target_name):
    print(f"\nGenerating SHAP Explainability for {target_name}...")
    model = models[target_name]
    
    # Create the SHAP Explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    print("SHAP values calculated successfully. (These will be used in the Django UI for 'i' tooltips).")
    # In a notebook, run: shap.summary_plot(shap_values, X_test)
    return explainer

# Exporting for Django Deployment
def export_models(models):
    print("\nExporting models for Django deployment...")
    joblib.dump(models, 'sayardesk_ielts_models.joblib')
    print("Successfully saved 'sayardesk_ielts_models.joblib'.")


if __name__ == "__main__":
    file1 = 'D:\Projects\Senior-Capstone-Project\Datasets\Scraping Data\combined_ielts_essays_fixed.csv'
    file2 = 'D:\Projects\Senior-Capstone-Project\Datasets\AI Generated Dataset\ielts_gemini_fixed.csv'
    
    try:
        df, targets = load_and_clean_data(file1, file2)

        X, handcrafted_cols = engineer_features(df)
        
        trained_models, X_train, X_test = train_evaluate_save_models(X, df, targets)
        
        # Explainability Check (Testing on Lexical Resource as an example)
        # This proves the "Human-in-the-loop" transparency requirement
        generate_shap_explanations(trained_models, X_test, 'Lexical_Resource')
        
        export_models(trained_models)
        
        print("\nPipeline Complete!")
        
    except FileNotFoundError:
        print("Error: Could not find the CSV files.")