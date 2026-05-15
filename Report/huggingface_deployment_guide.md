# HuggingFace Spaces Deployment Guide
### SayarDesk IELTS Intelligence — Complete Workflow

---

## 🏗️ How The Architecture Works

```
Your Local PC
    │
    │  git push
    ▼
GitHub (Thriple-T/Senior-Capstone-Project)
    │
    │  GitHub Actions triggers automatically
    │  (sync_to_hf.yml)
    ▼
HuggingFace Space (ThripleT1912/SayarDesk-IELTS-Intelligence)
    │
    │  HF detects new files → rebuilds Docker container
    ▼
Live App at: https://thriplet1912-sayardesk-ielts-intelligence.hf.space
```

**Key Concept:** You never push directly to HuggingFace. You push to GitHub, and a GitHub Action automatically copies the right files to HF for you.

---

## 🔧 One-Time Setup (Already Done for Your Project)

These steps only need to be done once. They are already configured for your project, but it's useful to understand them.

### Step 1: Create a HuggingFace Space

1. Go to [huggingface.co](https://huggingface.co) and log in.
2. Click **+ New Space**.
3. Choose:
   - **Owner:** `ThripleT1912`
   - **Space name:** `SayarDesk-IELTS-Intelligence`
   - **SDK:** `Docker` ← Critical! This lets you use your `Dockerfile`.
   - **Visibility:** Public or Private
4. Click **Create Space**.

> ✅ Your space is already created at `ThripleT1912/SayarDesk-IELTS-Intelligence`

---

### Step 2: Get Your HuggingFace Token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
2. Click **New token**.
3. Name it (e.g., `GitHub Actions`).
4. Set **Role** to `Write`.
5. Copy the token — it looks like `hf_xxxxxxxxxxxxxxxxxx`.

> ⚠️ Save this somewhere safe. You can't see it again after closing the page.

---

### Step 3: Add the HF Token to GitHub Secrets

1. Go to your GitHub repo: `github.com/Thriple-T/Senior-Capstone-Project`
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Name: `HF_TOKEN`
5. Value: Paste your token from Step 2
6. Click **Add secret**

> ✅ This is already set up. The `sync_to_hf.yml` references it as `${{ secrets.HF_TOKEN }}`.

---

### Step 4: Add App Secrets to HuggingFace Space

Your app needs environment variables (like API keys) at runtime. Add them in HuggingFace, NOT in your code.

1. Go to your Space: [huggingface.co/spaces/ThripleT1912/SayarDesk-IELTS-Intelligence](https://huggingface.co/spaces/ThripleT1912/SayarDesk-IELTS-Intelligence)
2. Click **Settings** tab
3. Scroll to **Variables and Secrets**
4. Click **New secret** for each of these:

| Secret Name | What It Is |
|---|---|
| `SECRET_KEY` | Your Django secret key from `.env` |
| `DEBUG` | Set to `False` for production |
| `OPENAI_API_KEY` | Your OpenAI / LLM API key |
| `DB_HOST` | (Optional) PostgreSQL host if using external DB |

> ⚠️ These are **injected as environment variables** at runtime. Your `settings.py` reads them automatically via `python-decouple`.

---

## 🚀 Day-to-Day Development Workflow

This is what you do every day when you make changes.

### Making a Code Change

```
1. Edit your files locally in VS Code
2. Test locally:   python manage.py runserver
3. When happy:
   git add .
   git commit -m "Your description of what changed"
   git push
4. Wait ~5-10 minutes for HF to rebuild
5. Check your live Space URL
```

That's it. The GitHub Action handles everything in between.

---

### Monitoring the Deployment

**Watch the GitHub Action run:**
1. Go to your GitHub repo
2. Click the **Actions** tab
3. You'll see "Sync to HuggingFace Space" running
4. Click it to see live logs — this is where you'll catch errors

**Watch the HuggingFace build:**
1. Go to your HF Space
2. The status badge at the top will show:
   - 🟡 **Building** — Docker container is being built
   - 🟢 **Running** — App is live
   - 🔴 **Error** — Something went wrong (click **Logs** to see why)

---

## 📂 What Gets Synced — and Why

Your `sync_to_hf.yml` uploads these files to HuggingFace:

| File/Folder | Purpose |
|---|---|
| `Dockerfile` | Tells HF how to build and run your app |
| `requirements.txt` | Python packages to install |
| `manage.py` | Django entry point |
| `sayardesk/` | Django project config (settings, urls, wsgi) |
| `ielts_engine/` | Your main app (views, models, ML logic) |
| `templates/` | HTML templates |
| `static/` | CSS, JS, images |
| `sayardesk_model.pth` | BERT model weights (~438 MB) |
| `hybrid_xgb_model.joblib` | XGBoost model |
| `ielts_scaler.pkl` | Feature scaler |

> **Why not just sync everything?** Your GitHub repo contains large datasets, notebooks, reports, and your `.venv` that HF doesn't need. Selective upload keeps the HF Space lean and fast to build.

---

## 🐛 Troubleshooting Common Issues

### "No module named 'torch'" or AI Engine Fails
**Cause:** `requirements.txt` has version numbers that don't exist on PyPI.
**Fix:** Use `>=` instead of `==` for ML libraries.
```
# Bad (version might not exist publicly)
torch==2.10.0

# Good (installs latest stable)
torch>=2.1.0
```

### Space is "Running" but shows an error page
**Fix:** Click **Logs** on your HF Space → look for the actual Python error.

### GitHub Action fails
**Fix:** Go to GitHub → Actions tab → click the failed run → read the error message.

### Model files aren't being uploaded (GitHub LFS issue)
**Cause:** Large files (`.pth`, `.joblib`) are stored in Git LFS. The GitHub Action needs to check them out properly.
**Fix:** Add `lfs: true` to the checkout step in `sync_to_hf.yml`:
```yaml
- uses: actions/checkout@v3
  with:
    lfs: true
```

### Changes not appearing after push
1. Check the GitHub Actions tab — did the workflow run?
2. Check the HF Space — is it still "Building"?
3. Hard-refresh your browser on the HF Space URL (Ctrl+Shift+R)

---

## 🔄 Current `sync_to_hf.yml` Explained Line by Line

```yaml
name: Sync to HuggingFace Space   # Name shown in GitHub Actions tab

on:
  push:
    branches: [main]               # Trigger ONLY when you push to main branch

jobs:
  sync:
    runs-on: ubuntu-latest         # GitHub spins up a fresh Linux machine

    steps:
      - uses: actions/checkout@v3  # Downloads your repo code onto that machine

      - name: Install HuggingFace Hub + Xet
        run: pip install -U huggingface_hub hf_xet   # Install HF upload tools

      - name: Upload app code and artifacts to HuggingFace
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}   # Reads your secret token
        run: |
          huggingface-cli login --token $HF_TOKEN   # Authenticate with HF

          # Each line uploads one file/folder to your HF Space
          huggingface-cli upload [SPACE_ID] [LOCAL_PATH] [REMOTE_PATH] --repo-type space
```

---

## 💡 Tips for Your Project

1. **Test locally first** — always run `python manage.py runserver` before pushing
2. **Commit often** — small commits are easier to debug if something breaks
3. **Watch the logs** — the HF build logs tell you exactly what went wrong
4. **Model files are big** — uploading `sayardesk_model.pth` (438 MB) takes a few minutes
5. **Secrets stay secret** — never put API keys in your code; always use HF Secrets
