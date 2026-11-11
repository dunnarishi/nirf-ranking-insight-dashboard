# NIRF Ranking Insight Dashboard

Python-only interactive dashboard for NIRF rankings from 2017 to 2025.

## Run the project

1. Install Python 3.10 or newer.
2. Install dependencies:

   ```powershell
   py -m pip install -r requirements.txt
   ```

3. Start the dashboard:

   ```powershell
   py -m streamlit run app.py
   ```

Streamlit will open the dashboard in a browser. The application automatically reads `nirf_raw.csv` from the same folder.

## Project files

- `app.py` - Python dashboard, data preparation, 40 Level 1 analyses, dynamic Level 2 analysis, and Level 3 natural-language query handling.
- `nirf_raw.csv` - supplied NIRF dataset.
- `requirements.txt` - Python dependencies.

## Features

- Filters for category, state, institution, analysis year, and comparison years.
- All eight Annexure A analysis groups, including trend, correlation, multivariate, mobility, and outlier analysis.
- Configurable charts for new evaluation questions.
- Natural-language input for common questions about years, states, ranks, and parameter scores.

## Data handling

The app standardizes source column names, converts rank and score fields to numbers, normalizes whitespace in text fields, preserves missing parameter values, and removes duplicate institution/category/year records. Ranks are lower-is-better.
