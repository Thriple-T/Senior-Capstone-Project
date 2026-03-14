import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from advanced_model import IELTSMultiTaskModel, EssayDataset, extract_advanced_features
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import joblib

def run_assessment(csv_path, model_path=r'D:\Projects\Senior-Capstone-Project\ML Models\sayardesk_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv(csv_path).dropna()
    
    # Re-extract features (same as training)
    print("Extracting features for assessment...")
    feats = np.array([extract_advanced_features(row) for _, row in df.iterrows()])
    
    try:
        scaler = joblib.load(r'D:\Projects\Senior-Capstone-Project\ML Models\ielts_scaler.pkl')
        feats = scaler.transform(feats)
    except FileNotFoundError:
        print("Warning: ielts_scaler.pkl not found! Using raw features.")
        
    targets = df[['Task_Achievement', 'Coherence_Cohesion', 'Lexical_Resource', 'Grammar_Range']].values
    
    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = EssayDataset(df['Essay_Text'].values, feats, targets, tok)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    model = IELTSMultiTaskModel(n_extra_features=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    try:
        xgb_model = joblib.load(r'D:\Projects\Senior-Capstone-Project\ML Models\hybrid_xgb_model.joblib')
        has_xgb = True
    except FileNotFoundError:
        print("Warning: hybrid_xgb_model.joblib not found. The model will fall back to base PyTorch outputs.")
        has_xgb = False
    
    all_preds = []
    all_actuals = []

    print("Running model inference...")
    with torch.no_grad():
        for batch in loader:
            ids, mask = batch['ids'].to(device), batch['mask'].to(device)
            f, t = batch['feats'].to(device), batch['targets'].to(device)
            
            (ta, cc, lr, gra), hybrid_feats = model(ids, mask, f)
            
            if has_xgb:
                xgb_preds = xgb_model.predict(hybrid_feats.cpu().numpy())
                all_preds.append(xgb_preds)
            else:
                combined_preds = torch.stack([ta.squeeze(), cc.squeeze(), lr.squeeze(), gra.squeeze()], dim=1)
                all_preds.append(combined_preds.cpu().numpy())
            all_actuals.append(t.cpu().numpy())

    # Calculating Metrics
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_actuals)
    criteria = ['Task Achievement', 'Coherence', 'Lexical', 'Grammar']
    
    report = []
    for i, name in enumerate(criteria):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        
        # Custom IELTS accuracy: Within 0.5 band
        diff = np.abs(np.round(y_pred[:, i] * 2) / 2 - y_true[:, i])
        within_half = (diff <= 0.5).mean() * 100
        
        report.append({
            "Criteria": name,
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "R2": round(r2, 3),
            "Within 0.5 Band (%)": f"{within_half:.1f}%"
        })

    return pd.DataFrame(report)

if __name__ == "__main__":
    # Use test dataset file here
    results_df = run_assessment(r'D:\Projects\Senior-Capstone-Project\Datasets\Scraping Data\combined_ielts_essays_fixed.csv')
    print("\n MODEL ASSESSMENT REPORT")
    print(results_df.to_string(index=False))
    