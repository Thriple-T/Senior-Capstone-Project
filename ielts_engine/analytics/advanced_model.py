import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from torch.optim import AdamW  # Don't forget this
from torch.optim.lr_scheduler import OneCycleLR
from sentence_transformers import SentenceTransformer, util
import spacy
from spellchecker import SpellChecker
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import joblib
import json
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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
    misspelled = len(spell.unknown(tokens)) / len(tokens) if tokens else 0
    
    # Task Achievement: Prompt Similarity
    sim = 0.0
    if essay and prompt:
        # Using encoding without tensor conversion here to avoid device mismatch during extraction
        e_emb = sim_model.encode(essay, convert_to_numpy=True)
        p_emb = sim_model.encode(prompt, convert_to_numpy=True)
        sim = util.cos_sim(e_emb, p_emb).item()
        
    return [float(var), float(dep_clauses), float(ttr), float(misspelled), float(sim)]

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
    joblib.dump(scaler, 'ielts_scaler.pkl')
    print("Scaler saved as ielts_scaler.pkl")
    
    targets = df[['Task_Achievement', 'Coherence_Cohesion', 'Lexical_Resource', 'Grammar_Range']].values
    
    # 3-way split: 64% train / 16% val / 20% test
    train_txt, test_txt, train_f, test_f, train_y, test_y = train_test_split(
        df['Essay_Text'].values, feats, targets, test_size=0.2, random_state=42
    )
    train_txt, val_txt, train_f, val_f, train_y, val_y = train_test_split(
        train_txt, train_f, train_y, test_size=0.2, random_state=42
    )
    
    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    train_loader = DataLoader(EssayDataset(train_txt, train_f, train_y, tok), batch_size=8, shuffle=True)
    val_loader = DataLoader(EssayDataset(val_txt, val_f, val_y, tok), batch_size=8, shuffle=False)
    
    NUM_EPOCHS = 5
    LR = 3e-5
    
    model = IELTSMultiTaskModel(n_extra_features=feats.shape[1]).to(device)
    opt = AdamW(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    
    # Warmup (10%) + linear decay scheduler
    scheduler = OneCycleLR(
        opt, max_lr=LR,
        epochs=NUM_EPOCHS, steps_per_epoch=len(train_loader),
        pct_start=0.1, anneal_strategy='linear'
    )
    
    # Training history for the analysis notebook
    history = {'train_loss': [], 'val_loss': [], 'lr_log': []}
    
    print(f"Starting training... (LR={LR}, Epochs={NUM_EPOCHS})")
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
            scheduler.step()
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

    torch.save(model.state_dict(), 'sayardesk_model.pth')
    print("Model saved as sayardesk_model.pth")
    
    # Save training history for analysis notebook
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print("Training history saved as training_history.json")
    
    # Hybrid Stage: BERT + XGBoost
    print("\nExtracting 773-dim hybrid features for XGBoost...")
    model.eval()
    all_hybrid_features = []
    all_targets = []
    
    with torch.no_grad():
        for batch in train_loader:
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            f = batch['feats'].to(device)
            t = batch['targets']
            
            _, hybrid_feats = model(ids, mask, f)
            all_hybrid_features.append(hybrid_feats.cpu().numpy())
            all_targets.append(t.numpy())
            
    X_train_xgb = np.vstack(all_hybrid_features)
    y_train_xgb = np.vstack(all_targets)
    
    print(f"Hybrid feature shape: {X_train_xgb.shape} (Expected: N x 773)")
    
    print("Training Hybrid XGBoost Model...")
    xgb_base = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    multi_xgb = MultiOutputRegressor(xgb_base)
    multi_xgb.fit(X_train_xgb, y_train_xgb)
    
    joblib.dump(multi_xgb, 'hybrid_xgb_model.joblib')
    print("Hybrid XGBoost Model saved as hybrid_xgb_model.joblib")

if __name__ == "__main__":
    run_pipeline()