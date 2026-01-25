# Deployment Guide - Streamlit Dashboard

## ⚠️ Important: Netlify Does NOT Support Streamlit

Netlify is for **static websites** (HTML/CSS/JS). Streamlit is a **Python web framework** and requires a Python runtime environment.

---

## ✅ Free Platforms That Support Streamlit

### 🥇 Option 1: Streamlit Cloud (RECOMMENDED - Easiest & Free)

**Why Choose This:**
- ✅ 100% Free (no credit card needed)
- ✅ Easiest setup (just connect GitHub)
- ✅ Automatic deployments on git push
- ✅ Built specifically for Streamlit apps
- ✅ Custom domain support

**Setup Steps:**

1. **Prepare Your GitHub Repo:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "Sign in" → Sign in with GitHub
   - Click "New app"
   - Fill in:
     - **Repository:** Select your GitHub repo
     - **Branch:** `main` (or `master`)
     - **Main file path:** `app.py`
     - **App URL:** (auto-generated, or choose custom)
   - Click "Deploy"
   - Wait 2-3 minutes for deployment

3. **Your app will be live at:** `https://your-app-name.streamlit.app`

**Requirements:**
- ✅ `app.py` in root directory
- ✅ `requirements.txt` with all dependencies
- ✅ `processed_fleet_data.csv` (or sample data) in repo

**Note:** If you don't want to commit `processed_fleet_data.csv`, you can:
- Use Streamlit's file uploader to upload CSV
- Or provide sample data in the repo

---

### 🥈 Option 2: Railway (Free Tier Available)

**Why Choose This:**
- ✅ Free tier: $5 credit/month (enough for small apps)
- ✅ Easy deployment
- ✅ Supports Python/Streamlit

**Setup Steps:**

1. **Create `railway.json` (already created for you)**

2. **Deploy:**
   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repo
   - Railway auto-detects Python and deploys

3. **Set Environment Variables (if needed):**
   - Go to project settings
   - Add any required env vars

**Your app will be live at:** `https://your-app-name.up.railway.app`

---

### 🥉 Option 3: Render (Free Tier Available)

**Why Choose This:**
- ✅ Free tier available
- ✅ Auto-deploy from GitHub
- ✅ Easy setup

**Setup Steps:**

1. **Create `render.yaml` (already created for you)**

2. **Deploy:**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub
   - Click "New" → "Web Service"
   - Connect your GitHub repo
   - Settings:
     - **Name:** Your app name
     - **Environment:** Python 3
     - **Build Command:** `pip install -r requirements.txt`
     - **Start Command:** `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
   - Click "Create Web Service"

**Your app will be live at:** `https://your-app-name.onrender.com`

**Note:** Free tier apps sleep after 15 minutes of inactivity (takes ~30 seconds to wake up)

---

### Option 4: Fly.io (Free Tier Available)

**Why Choose This:**
- ✅ Free tier: 3 shared VMs
- ✅ Global edge network
- ✅ Good performance

**Setup Steps:**

1. **Install Fly CLI:**
   ```bash
   # Windows (PowerShell)
   iwr https://fly.io/install.ps1 -useb | iex
   ```

2. **Login:**
   ```bash
   fly auth login
   ```

3. **Deploy:**
   ```bash
   fly launch
   ```
   - Follow prompts
   - Select region
   - Deploy!

**Your app will be live at:** `https://your-app-name.fly.dev`

---

## 📋 Comparison Table

| Platform | Free Tier | Ease of Setup | Auto-Deploy | Best For |
|----------|-----------|---------------|-------------|----------|
| **Streamlit Cloud** | ✅ Yes | ⭐⭐⭐⭐⭐ | ✅ Yes | **Best choice** |
| **Railway** | ✅ $5/month credit | ⭐⭐⭐⭐ | ✅ Yes | Good alternative |
| **Render** | ✅ Yes (sleeps) | ⭐⭐⭐⭐ | ✅ Yes | Budget option |
| **Fly.io** | ✅ 3 VMs | ⭐⭐⭐ | ⚠️ Manual | Advanced users |

---

## 🚀 Recommended: Streamlit Cloud

**Why?**
- Built specifically for Streamlit
- Zero configuration needed
- Free forever
- Best performance for Streamlit apps

**Quick Start:**
1. Push code to GitHub
2. Go to share.streamlit.io
3. Connect repo → Deploy
4. Done! 🎉

---

## 📝 Files Needed for Deployment

All these files are already created:

- ✅ `requirements.txt` - Python dependencies
- ✅ `Procfile` - For Heroku/Railway
- ✅ `runtime.txt` - Python version
- ✅ `setup.sh` - Streamlit config
- ✅ `.gitignore` - Excludes sensitive data

**Optional (for specific platforms):**
- `railway.json` - Railway config (created)
- `render.yaml` - Render config (created)

---

## 🔒 Security Note

Your `.gitignore` already excludes:
- `Asset_Module/` - Source Excel files (sensitive data)
- `venv/` - Virtual environment
- `PROJECT_EXPLANATION_FOR_MANAGER.md` - Internal docs

**For deployment, you need:**
- `processed_fleet_data.csv` - This should be in the repo (or use sample data)

---

## 🆘 Troubleshooting

### App won't start on Streamlit Cloud:
- Check `requirements.txt` has all dependencies
- Ensure `app.py` is in root directory
- Check logs in Streamlit Cloud dashboard

### App crashes:
- Verify `processed_fleet_data.csv` exists in repo
- Check Python version compatibility (3.11 recommended)
- Review error logs in platform dashboard

### Need help?
- Streamlit Cloud: [docs.streamlit.io](https://docs.streamlit.io/streamlit-community-cloud)
- Railway: [docs.railway.app](https://docs.railway.app)
- Render: [render.com/docs](https://render.com/docs)

---

**Recommendation: Use Streamlit Cloud - it's the easiest and best option for Streamlit apps!**

