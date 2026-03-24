import torch
import numpy as np
import joblib
import pandas as pd
import torch.nn as nn
import spacy
from spellchecker import SpellChecker
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer, util
import warnings

warnings.filterwarnings('ignore')

print("Loading NLP pipelines (this may take a moment)...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

spell = SpellChecker()
sim_model = SentenceTransformer('all-mpnet-base-v2')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class IELTSMultiTaskModel(nn.Module):
    def __init__(self, n_extra_features):
        super(IELTSMultiTaskModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(0.3)
        input_size = self.bert.config.hidden_size + n_extra_features
        
        self.ta_head = nn.Linear(input_size, 1)
        self.cc_head = nn.Linear(input_size, 1)
        self.lr_head = nn.Linear(input_size, 1)
        self.gra_head = nn.Linear(input_size, 1)

    def forward(self, ids, mask, feats):
        out = self.bert(ids, attention_mask=mask)
        pooled = self.drop(out.pooler_output)
        combined = torch.cat((pooled, feats), dim=1)
        return (self.ta_head(combined), self.cc_head(combined), \
               self.lr_head(combined), self.gra_head(combined)), combined

def extract_advanced_features(essay, prompt):
    doc = nlp(essay)
    
    # Grammar: Sentence Variance & Dependent Clauses
    lengths = [len(s) for s in doc.sents]
    var = np.var(lengths) if lengths else 0
    dep_clauses = sum(1 for t in doc if t.dep_ in ['advcl', 'relcl', 'ccomp'])
    
    # Lexical: Spell Errors & Diversity
    tokens = [t.text.lower() for t in doc if t.is_alpha]
    unique = set(tokens)
    ttr = len(unique) / len(tokens) if tokens else 0
    misspelled = len(spell.unknown(tokens)) / len(tokens) if tokens else 0
    
    # Task Achievement: Prompt Similarity
    sim = 0.0
    if essay and prompt:
        e_emb = sim_model.encode(essay, convert_to_numpy=True)
        p_emb = sim_model.encode(prompt, convert_to_numpy=True)
        sim = util.cos_sim(e_emb, p_emb).item()
        
    return [float(var), float(dep_clauses), float(ttr), float(misspelled), float(sim)]
import os
from django.conf import settings
from .llm_grader import get_llm_assessment

def predict_ensemble(essay_text, prompt_text, 
                     pytorch_model_path=None,
                     xgb_model_path=None,
                     scaler_path=None):
    
    # Use Django Base Dir dynamically if paths aren't provided
    if pytorch_model_path is None:
        pytorch_model_path = os.path.join(settings.BASE_DIR, 'ielts_engine', 'analytics', 'sayardesk_model.pth')
    if xgb_model_path is None:
        xgb_model_path = os.path.join(settings.BASE_DIR, 'ielts_engine', 'analytics', 'hybrid_xgb_model.joblib')
    if scaler_path is None:
        scaler_path = os.path.join(settings.BASE_DIR, 'ielts_engine', 'analytics', 'ielts_scaler.pkl')
                     
    feats = extract_advanced_features(essay_text, prompt_text)
    try:
        scaler = joblib.load(scaler_path)
        feats = scaler.transform([feats])[0]
    except FileNotFoundError:
        print(f"Warning: {scaler_path} not found. Ensure advanced_model.py has been run.")

    feats_tensor = torch.tensor([feats], dtype=torch.float).to(device)
    
    enc = tokenizer(
        essay_text, 
        max_length=512, 
        padding='max_length', 
        truncation=True, 
        return_tensors='pt'
    )
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)

    # Hybrid Stage (BERT Feature Extraction -> XGBoost Regressor)
    pt_model = IELTSMultiTaskModel(n_extra_features=5).to(device)
    pt_model.load_state_dict(torch.load(pytorch_model_path, map_location=device))
    pt_model.eval()

    with torch.no_grad():
        _, hybrid_feats = pt_model(input_ids, attention_mask, feats_tensor)
    
    hybrid_vector = hybrid_feats.cpu().numpy() # Shape: (1, 773)
    
    # Load Hybrid Model
    xgb_model = joblib.load(xgb_model_path)
    hybrid_preds = xgb_model.predict(hybrid_vector)[0]
    
    # --- SHAP Explainability ---
    shap_summary = []
    try:
        import shap
        
        feature_names = ["Semantic Meaning", "Sentence Variance", "Dependent Clauses", 
                         "Lexical Diversity (TTR)", "Misspelled Ratio", "Prompt Similarity"]
                         
        if hasattr(xgb_model, 'estimators_'):
            # MultiOutputRegressor: extract from each underlying estimator and average
            all_shap_values = []
            for estimator in xgb_model.estimators_:
                explainer = shap.TreeExplainer(estimator)
                all_shap_values.append(explainer.shap_values(hybrid_vector))
            
            # Average the SHAP values across all 4 estimators
            # shap_values from each estimator is usually (1, 773)
            # all_shap_values will be shape (4, 1, 773)
            avg_shap = np.mean(all_shap_values, axis=0) # shape (1, 773)
            
            if len(avg_shap.shape) == 3:
                sv_signed = avg_shap[0].mean(axis=1)
            else:
                sv_signed = avg_shap[0]
        else:
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(hybrid_vector)
            
            if isinstance(shap_values, list):
                sv_signed = np.array(shap_values).mean(axis=0)[0]
            elif len(shap_values.shape) == 3:
                sv_signed = shap_values[0].mean(axis=1)
            else:
                sv_signed = shap_values[0]
            
        bert_val = float(np.sum(sv_signed[:768]))
        adv_vals = [float(x) for x in sv_signed[768:]]
        all_vals = [bert_val] + adv_vals
        
        for name, val in zip(feature_names, all_vals):
            shap_summary.append({
                "feature": name,
                "impact": val,
                "type": "positive" if val >= 0 else "negative",
                "abs_impact": abs(val)
            })
            
        shap_summary.sort(key=lambda x: x["abs_impact"], reverse=True)
        shap_summary = shap_summary[:5]
    except Exception as e:
        print(f"SHAP Error: {e}")

    hybrid_scores = {
        "Task_Achievement": hybrid_preds[0],
        "Coherence_Cohesion": hybrid_preds[1],
        "Lexical_Resource": hybrid_preds[2],
        "Grammar_Range": hybrid_preds[3]
    }

    # LLM Stage (Qwen3-VL-8B-Instruct) with Fallback in case of API Failure
    try:
        llm_assessment = get_llm_assessment(prompt_text, essay_text)
        llm_scores_dict = getattr(llm_assessment, "model_dump", lambda: llm_assessment.dict())()
        llm_available = True
    except Exception as e:
        print(f"\n[WARNING] LLM Connection Failed ({e}). Falling back to 100% XGBoost Hybrid Scoring.")
        llm_available = False
        llm_scores_dict = {
            "Task_Achievement": "N/A (Offline)",
            "Coherence_Cohesion": "N/A (Offline)",
            "Lexical_Resource": "N/A (Offline)",
            "Grammar_Range": "N/A (Offline)",
            "total_band": "N/A",
            "feedback_summary": "LLM Engine is currently unavailable. Scores provided are purely based on the Hybrid Machine Learning model."
        }

    # Ensemble Stage (Weighted Average)
    ensemble_scores = {}
    
    if llm_available:
        # Custom Weights: (Hybrid_Weight, LLM_Weight)
        # XGBoost is better at Grammar/Lexical word-counts. LLM is better at Meaning/Task.
        weights = {
            "Task_Achievement": (0.3, 0.7),
            "Coherence_Cohesion": (0.4, 0.6),
            "Lexical_Resource": (0.7, 0.3),
            "Grammar_Range": (0.8, 0.2)
        }
        
        for crit, (w_hybrid, w_llm) in weights.items():
            h_score = hybrid_scores[crit]
            l_score = llm_scores_dict[crit]
            ens_raw = (h_score * w_hybrid) + (l_score * w_llm)
            ensemble_scores[crit] = round(ens_raw * 2) / 2 # Round to nearest IELTS band
            
    else:
        # Fallback to 100% Hybrid Model
        for crit in ["Task_Achievement", "Coherence_Cohesion", "Lexical_Resource", "Grammar_Range"]:
            ensemble_scores[crit] = round(hybrid_scores[crit] * 2) / 2
            
    overall_band = sum(ensemble_scores.values()) / 4
    ensemble_scores['Overall_Band'] = round(overall_band * 2) / 2
    
    return {
        "hybrid_scores": hybrid_scores,
        "llm_scores": llm_scores_dict,
        "ensemble_scores": ensemble_scores,
        "shap_data": shap_summary
    }

if __name__ == "__main__":
    # Some bad essay
    sample_prompt = "Some people believe that university education should be free for everyone. To what extent do you agree or disagree?"
    sample_essay = "The debate regarding the accessibility of higher education is a highly contentious one. While some argue that university tuition should be entirely subsidized by the government, I firmly believe that a completely free system is economically unsustainable and could lead to a devaluation of the degrees themselves."
    print("Running Master Ensemble Predictor...\n")
    try:
        results = predict_ensemble(sample_essay, sample_prompt)
        print("Hybrid Model (BERT+XGB) Scores")
        for k, v in results['hybrid_scores'].items(): print(f"  {k}: {v:.2f}")
        
        print("\nLLM (Qwen3-VL) Scores")
        llm = results['llm_scores']
        for k in ["Task_Achievement", "Coherence_Cohesion", "Lexical_Resource", "Grammar_Range", "total_band"]:
            print(f"  {k}: {llm[k]}")
            
        print("\n Final Ensemble Scores (Weighted Average)")
        for k, v in results['ensemble_scores'].items(): print(f"  {k}: {v:.1f}")
        
        print("\nLLM Feedback")
        print(llm['feedback_summary'])
    except Exception as e:
        print(f"Error during prediction: {e}")
