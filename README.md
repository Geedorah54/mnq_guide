# MNQ Options Flow Trading Dashboard

This project is a Streamlit-based trading assistant for Micro Nasdaq Futures (MNQ).  
It integrates:

- EMA trend model (8/21)
- VIX volatility regime detection
- Gamma exposure (dealer positioning)
- Options chain analysis (call wall, put wall, OI clusters)
- Trend score (0–100)
- A+ setup checklist
- Real-time charts and market conditions

## Running the App

```bash
# activate your venv
source venv/bin/activate

# install dependencies
pip install -r requirements.txt

# run the app
streamlit run app/app.py
