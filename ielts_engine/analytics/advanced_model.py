import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from torch.optim import AdamW  # Don't forget this
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sentence_transformers import SentenceTransformer, util
import spacy
from spellchecker import SpellChecker
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import joblib
import json
import warnings
import textstat

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Path logic for saving models to Root
import os
try:
    from django.conf import settings
    BASE_PATH = settings.BASE_DIR
except:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    BASE_PATH = os.path.abspath(os.path.join(current_dir, '..', '..'))

try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

spell = SpellChecker()
sim_model = SentenceTransformer('all-mpnet-base-v2')

def extract_advanced_features(row):
    essay = str(row.get('Essay_Text', ''))
    prompt = str(row.get('Question', ''))
    doc = nlp(essay)
    
    # Grammar: Sentence Variance & Dependent Clauses
    lengths = [len(s) for s in doc.sents]
    var = np.var(lengths) if lengths else 0
    dep_clauses = sum(1 for t in doc if t.dep_ in ['advcl', 'relcl', 'ccomp'])
    
    # Lexical: Spell Errors & Diversity
    tokens = [t.text.lower() for t in doc if t.is_alpha]
    unique = set(tokens)
    ttr = len(unique) / len(tokens) if tokens else 0
    misspelled_count = len(spell.unknown(tokens)) if tokens else 0
    error_density = (misspelled_count / len(tokens)) * 100 if tokens else 0
    
    # Task Achievement: Prompt Similarity
    sim = 0.0
    if essay and prompt:
        # Using encoding without tensor conversion here to avoid device mismatch during extraction
        e_emb = sim_model.encode(essay, convert_to_numpy=True)
        p_emb = sim_model.encode(prompt, convert_to_numpy=True)
        sim = util.cos_sim(e_emb, p_emb).item()
        
    # Readability
    flesch = textstat.flesch_reading_ease(essay)
    
    # Cohesion Markers
    cohesion_words = {'however', 'furthermore', 'moreover', 'consequently', 'therefore', 'thus', 'hence', 'nevertheless', 'additionally', 'meanwhile'}
    cohesion = sum(1 for t in tokens if t in cohesion_words)
        
    return [float(var), float(dep_clauses), float(ttr), float(misspelled_count), float(sim), float(flesch), float(cohesion), float(error_density)]

class IELTSMultiTaskModel(nn.Module):
    def __init__(self, n_extra_features):
        super(IELTSMultiTaskModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(0.3)
        input_size = self.bert.config.hidden_size + n_extra_features
        
        # Shared base with 4 task-specific heads
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

# Customized Weighted Loss
def weighted_loss(preds, targets):
    # Penalize errors on extreme scores more heavily
    weights = torch.ones_like(targets)
    weights[(targets < 5.5) | (targets > 7.5)] = 2.5
    return (weights * (preds - targets)**2).mean()

class EssayDataset(Dataset):
    def __init__(self, texts, features, targets, tokenizer):
        self.texts = texts
        self.features = np.array(features, dtype=np.float32)
        self.targets = np.array(targets, dtype=np.float32)
        self.tokenizer = tokenizer

    def __len__(self): return len(self.texts)

    def __getitem__(self, i):
    # Use the tokenizer directly (this is the standard way according what I learnt so far, maybe change later???)
        enc = self.tokenizer(
            str(self.texts[i]), 
            max_length=512, 
            padding='max_length',
            truncation=True, 
            return_tensors='pt'
        )
        return {
            'ids': enc['input_ids'].flatten(),
            'mask': enc['attention_mask'].flatten(),
            'feats': torch.tensor(self.features[i], dtype=torch.float),
            'targets': torch.tensor(self.targets[i], dtype=torch.float)
        }

# Pipeline
def run_pipeline():
    print("Reading data...")
    df1 = pd.read_csv('D:\Projects\Senior-Capstone-Project\Datasets\Scraping Data\combined_ielts_essays_fixed.csv')
    df2 = pd.read_csv('D:\Projects\Senior-Capstone-Project\Datasets\AI Generated Dataset\ielts_gemini_fixed.csv')
    df = pd.concat([df1, df2]).dropna(subset=['Essay_Text', 'Question', 'Task_Achievement', 'Coherence_Cohesion', 'Lexical_Resource', 'Grammar_Range'])
    
    print("Processing linguistic features...")

    feature_list = df.apply(extract_advanced_features, axis=1).tolist()
    feats = np.array(feature_list)
    
    # Scale features for Neural Networks
    scaler = StandardScaler()
    feats = scaler.fit_transform(feats)
    joblib.dump(scaler, os.path.join(BASE_PATH, 'ielts_scaler.pkl'))
    print(f"Scaler saved to {os.path.join(BASE_PATH, 'ielts_scaler.pkl')}")
    
    targets = df[['Task_Achievement', 'Coherence_Cohesion', 'Lexical_Resource', 'Grammar_Range']].values
    
    # 3-way split: 64% train / 16% val / 20% test
    train_txt, test_txt, train_q, test_q, train_f, test_f, train_y, test_y = train_test_split(
        df['Essay_Text'].values, df['Question'].values, feats, targets,
        test_size=0.2, random_state=42
    )
    train_txt, val_txt, train_q, val_q, train_f, val_f, train_y, val_y = train_test_split(
        train_txt, train_q, train_f, train_y,
        test_size=0.2, random_state=42
    )

    # Save the test split so evaluate.py uses clean held-out data
    test_df = pd.DataFrame({
        'Essay_Text': test_txt,
        'Question':   test_q,
        'Task_Achievement':    test_y[:, 0],
        'Coherence_Cohesion':  test_y[:, 1],
        'Lexical_Resource':    test_y[:, 2],
        'Grammar_Range':       test_y[:, 3],
    })
    test_df.to_csv('test_split.csv', index=False)
    print(f"Test split saved: {len(test_df)} essays → test_split.csv")
    
    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    train_loader = DataLoader(EssayDataset(train_txt, train_f, train_y, tok), batch_size=8, shuffle=True)
    val_loader = DataLoader(EssayDataset(val_txt, val_f, val_y, tok), batch_size=8, shuffle=False)
    
    NUM_EPOCHS = 15
    LR = 3e-5
    
    model = IELTSMultiTaskModel(n_extra_features=feats.shape[1]).to(device)
    opt = AdamW(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)
    
    history = {'train_loss': [], 'val_loss': [], 'lr_log': []}
    
    print(f"Starting training... (LR={LR}, Epochs={NUM_EPOCHS})")
    
    best_val_loss = float('inf')
    patience_counter = 0
    EARLY_STOPPING_PATIENCE = 3
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            opt.zero_grad()
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            f = batch['feats'].to(device)
            t = batch['targets'].to(device)
            
            (ta, cc, lr, gra), _ = model(ids, mask, f)
            
            loss = loss_fn(ta.squeeze(), t[:, 0]) + \
                   loss_fn(cc.squeeze(), t[:, 1]) + \
                   loss_fn(lr.squeeze(), t[:, 2]) + \
                   loss_fn(gra.squeeze(), t[:, 3])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            epoch_loss += loss.item()
            history['lr_log'].append(opt.param_groups[0]['lr'])
        
        avg_train_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['ids'].to(device)
                mask = batch['mask'].to(device)
                f = batch['feats'].to(device)
                t = batch['targets'].to(device)
                (ta, cc, lr, gra), _ = model(ids, mask, f)
                loss = loss_fn(ta.squeeze(), t[:, 0]) + \
                       loss_fn(cc.squeeze(), t[:, 1]) + \
                       loss_fn(lr.squeeze(), t[:, 2]) + \
                       loss_fn(gra.squeeze(), t[:, 3])
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)
        
        # Early Stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'sayardesk_model_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load best model for the rest of the pipeline
    best_model_path = os.path.join(BASE_PATH, 'sayardesk_model_best.pth')
    try:
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model weights from {best_model_path}")
    except Exception as e:
        print(f"Could not load best model: {e}")

    final_model_path = os.path.join(BASE_PATH, 'sayardesk_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Model saved to {final_model_path}")
    
    # Save training history
    history_path = os.path.join(BASE_PATH, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    # Hybrid Stage: BERT + XGBoost
    print("\nExtracting 773-dim hybrid features for XGBoost...")
    model.eval()
    all_hybrid_features = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            f = batch['feats'].to(device)
            t = batch['targets']
            
            _, hybrid_feats = model(ids, mask, f)
            all_hybrid_features.append(hybrid_feats.cpu().numpy())
            all_targets.append(t.numpy())
            
    X_stack_xgb = np.vstack(all_hybrid_features)
    y_stack_xgb = np.vstack(all_targets)
    
    print(f"Hybrid feature shape: {X_stack_xgb.shape} (Expected: N x 773)")
    
    print("Training Hybrid XGBoost Model with GridSearchCV...")
    
    # We use a base regressor inside MultiOutputRegressor, but GridSearchCV can wrap it
    xgb_base = XGBRegressor(n_estimators=100, random_state=42)
    multi_xgb = MultiOutputRegressor(xgb_base)
    
    param_grid = {
        'estimator__max_depth': [3, 5, 7],
        'estimator__learning_rate': [0.01, 0.05, 0.1],
    }
    
    grid_search = GridSearchCV(multi_xgb, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search.fit(X_stack_xgb, y_stack_xgb)
    
    print(f"Best XGBoost Parameters: {grid_search.best_params_}")
    
    xgb_path = os.path.join(BASE_PATH, 'hybrid_xgb_model.joblib')
    joblib.dump(grid_search.best_estimator_, xgb_path)
    print(f"Hybrid XGBoost Model saved to {xgb_path}")

if __name__ == "__main__":
    run_pipeline()