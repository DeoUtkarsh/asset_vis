# Asset Failure Analytics - Bad Actor Leaderboard

A Python-based data pipeline and Streamlit dashboard for analyzing equipment failures and identifying bad actors in a maritime fleet.

## Project Structure

```
.
├── Asset_Module/              # Input Excel files folder
├── data_pipeline.py          # Data processing pipeline
├── app.py                    # Streamlit dashboard (to be created)
├── processed_fleet_data.csv # Output from data pipeline
├── requirements.txt          # Python dependencies
├── IMPLEMENTATION_PLAN.txt   # Detailed implementation plan
└── README.md                 # This file
```

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**
   - Place all Excel files in the `Asset_Module/` folder
   - Required files:
     - `Make & Model - Aux Engine -New.xlsx`
     - `Make & Model - Main Engine - New.xlsx`
     - `Make & Model - BWTS.xlsx`
     - `Incident Database.xlsx`
     - `Alert actions.xlsx`
     - `DG RMA Analysis.xlsm` (for spare parts data)
     - Other incident files (optional)

3. **Run Data Pipeline**
   ```bash
   python data_pipeline.py
   ```
   This will create `processed_fleet_data.csv` with calculated metrics.

4. **Run Dashboard**
   ```bash
   streamlit run app.py
   ```
   The dashboard will open in your browser at `http://localhost:8501`

## Deployment

### ⚠️ Important: Netlify Does NOT Support Streamlit

Netlify is for static websites only. Streamlit requires a Python runtime environment.

### ✅ Recommended: Streamlit Cloud (Free & Easiest)

**Why Choose This:**
- ✅ 100% Free (no credit card needed)
- ✅ Easiest setup (just connect GitHub)
- ✅ Built specifically for Streamlit apps
- ✅ Automatic deployments

**Quick Setup:**
1. Push your code to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. Deploy on Streamlit Cloud:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app" → Select your repo
   - Set Main file path: `app.py`
   - Click "Deploy"
   - Your app will be live in 2-3 minutes!

**Your app URL:** `https://your-app-name.streamlit.app`

### Alternative Free Platforms

1. **Railway** - Free $5/month credit
   - Go to [railway.app](https://railway.app)
   - Deploy from GitHub (auto-detects Python)
   - Uses `railway.json` config (already created)

2. **Render** - Free tier (apps sleep after inactivity)
   - Go to [render.com](https://render.com)
   - Deploy from GitHub
   - Uses `render.yaml` config (already created)

3. **Fly.io** - Free tier (3 shared VMs)
   - Install Fly CLI and run `fly launch`
   - Good for advanced users

**See `DEPLOYMENT_GUIDE.md` for detailed instructions for all platforms.**

## Features

### Data Pipeline (`data_pipeline.py`)
- ✅ Column name standardization
- ✅ Date format handling
- ✅ Master equipment table creation
- ✅ Master incidents table creation
- ✅ Master actions table creation
- ✅ **Master parts table creation** (from DG RMA Analysis.xlsm)
- ✅ Metric calculations (MTBF, MTBR, MTTR, failure rates)
- ✅ **Spare parts integration** (cost calculations, stock risk assessment)
- ✅ Bad actor identification per Make/Model

### Dashboard (`app.py`) - ✅ Ready!
- ✅ Level 1: Make/Model Selection with summary metrics (includes Trend & Cost Impact)
- ✅ Level 2: Top 10 Bad Actors (Components, Failure Modes, **Spare Parts** tabs)
- ✅ Level 3: Detailed Analytics with inline expanders
  - Component Analytics (MTBF, MTBR, MTTR)
  - Failure Mode & Root Cause Analysis
  - **Action Taken Analytics** (recurrence rates, MTBF before/after)
  - **Spare Parts Impact** (cost per failure, parts per failure, stock risk, top parts)
- ✅ Decision & Action Panel

## Implementation Plan

See `IMPLEMENTATION_PLAN.txt` for detailed specifications.

