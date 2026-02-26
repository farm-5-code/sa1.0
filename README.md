# Sport Analyzer

Multi-factor sports match analytics (Poisson / ELO / Dixon–Coles / ML) with optional weather + news context and a Streamlit dashboard.

## Quickstart

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure

Copy env template and set API keys:

```bash
cp sport_analyzer/.env.example .env
```

### 3) Run CLI

```bash
python -m sport_analyzer.main --matches
python -m sport_analyzer.main --home "Arsenal" --away "Chelsea" --date "2026-03-01T20:00:00" --city "London"
```

### 4) Run Dashboard

```bash
streamlit run app.py
```

## Packaging

You can also install as a package:

```bash
pip install .
```

Then run:

```bash
sport-analyzer --matches
```
