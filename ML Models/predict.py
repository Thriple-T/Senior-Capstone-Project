import torch
import torch.nn as nn
import numpy as np
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
sim_model = SentenceTransformer('all-MiniLM-L6-v2')
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
        return self.ta_head(combined), self.cc_head(combined), \
               self.lr_head(combined), self.gra_head(combined)

def extract_advanced_features(essay, prompt):
    doc = nlp(essay)
    
    # Grammar
    lengths = [len(s) for s in doc.sents]
    var = np.var(lengths) if lengths else 0
    dep_clauses = sum(1 for t in doc if t.dep_ in ['advcl', 'relcl', 'ccomp'])
    
    # Lexical
    tokens = [t.text.lower() for t in doc if t.is_alpha]
    unique = set(tokens)
    ttr = len(unique) / len(tokens) if tokens else 0
    misspelled = len(spell.unknown(tokens)) / len(tokens) if tokens else 0
    
    # Task Achievement (Similarity to prompt)
    sim = 0.0
    if essay and prompt:
        e_emb = sim_model.encode(essay, convert_to_numpy=True)
        p_emb = sim_model.encode(prompt, convert_to_numpy=True)
        sim = util.cos_sim(e_emb, p_emb).item()
        
    return [float(var), float(dep_clauses), float(ttr), float(misspelled), float(sim)]

# Inference Function
def predict_ielts_score(essay_text, prompt_text, model_path='D:\Projects\Senior-Capstone-Project\ML Models\sayardesk_model.pth'):
    # Initialize model with 5 extra features
    model = IELTSMultiTaskModel(n_extra_features=5).to(device)
    
    # Load the trained weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Set to evaluation mode (turns off dropout)
    except FileNotFoundError:
        print(f"Error: Could not find {model_path}. Please check the file path.")
        return None

    # Extract hand-crafted features
    feats = extract_advanced_features(essay_text, prompt_text)
    
    # NOTE: In a production environment, MUST load the saved StandardScaler here
    # and transform the features: feats = saved_scaler.transform([feats])[0]
    feats_tensor = torch.tensor([feats], dtype=torch.float).to(device)

    # Tokenize the essay (Using the corrected, stable method)
    enc = tokenizer(
        essay_text, 
        max_length=512, 
        padding='max_length', 
        truncation=True, 
        return_tensors='pt'
    )
    
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)

    with torch.no_grad(): # Disable gradient calculation for faster inference
        ta, cc, lr, gra = model(input_ids, attention_mask, feats_tensor)

    # Format Output
    scores = {
        "Task Achievement": round(ta.item(), 1),
        "Coherence & Cohesion": round(cc.item(), 1),
        "Lexical Resource": round(lr.item(), 1),
        "Grammatical Range & Accuracy": round(gra.item(), 1),
        "Overall Band": round((ta.item() + cc.item() + lr.item() + gra.item()) / 4 * 2) / 2 # IELTS rounds to nearest 0.5
    }
    
    return scores

# Test run with a sample essay and prompt
if __name__ == "__main__":
    sample_prompt = "Some people believe that university education should be free for everyone. To what extent do you agree or disagree?"
    sample_essay = """
    The debate regarding the accessibility of higher education is a highly contentious one. While some argue that university tuition should be entirely subsidized by the government, I firmly believe that a completely free system is economically unsustainable and could lead to a devaluation of the degrees themselves.
    
    Firstly, abolishing tuition fees would place an immense burden on taxpayers. Universities require substantial funding to maintain state-of-the-art facilities, attract top-tier professors, and fund groundbreaking research. If the financial responsibility shifts entirely to the state, it inevitably leads to increased taxation or detrimental cuts to other essential public services such as healthcare and infrastructure.
    
    Furthermore, when education is entirely free, there is a risk that students may not value it as highly. Financial investment often correlates with personal commitment. When students or their families are required to contribute, it fosters a sense of responsibility and encourages learners to maximize the return on their investment by studying diligently.
    """
    
    print("\nAnalyzing essay...")
    results = predict_ielts_score(sample_essay, sample_prompt)
    print("\n--- IELTS Predicted Band Scores ---")
    for criteria, score in results.items():
        print(f"{criteria}: {score}")