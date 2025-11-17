# Milestone 2: Feature Engineering, EDA & Evidently - Summary

## ✅ Completed Deliverables

### 1. Feature Engineering Pipeline
- **`features/featurizer.py`**: Kafka consumer that computes windowed features
  - Computes midprice returns (1s, 5s, 30s, 60s intervals)
  - Calculates rolling volatility (std of returns)
  - Computes bid-ask spread (absolute and relative)
  - Calculates trade intensity (trades per second)
  - Computes order-book imbalance
  - Outputs to Kafka topic `ticks.features`
  - Saves to Parquet format in `data/processed/`

### 2. Replay Script
- **`scripts/replay.py`**: Regenerates features from saved raw data
  - Processes NDJSON files identically to live consumer
  - Ensures deterministic feature computation
  - Outputs to `features_replay.parquet` and appends to main features file

### 3. EDA Notebook
- **`notebooks/eda.ipynb`**: Comprehensive exploratory data analysis
  - Loads and explores features
  - Computes future volatility (target variable)
  - Generates distribution plots (histogram, box plot, percentile plot)
  - Threshold selection with multiple options
  - Label creation based on selected threshold
  - Feature correlation analysis

### 4. Evidently Report Generator
- **`scripts/generate_evidently_report.py`**: Creates data quality and drift reports
  - Compares early vs late windows of data
  - Generates HTML and JSON reports
  - Includes data quality and drift metrics

### 5. Feature Specification Document
- **`docs/feature_spec.md`**: Complete specification including:
  - Target horizon: 60 seconds
  - Volatility proxy: rolling std of midprice returns
  - Label definition: binary classification based on threshold
  - Threshold selection methodology
  - Feature engineering details

### 6. Configuration Updates
- Updated `config.yaml` with feature engineering settings:
  - Window size: 300 seconds (5 minutes)
  - Prediction horizon: 60 seconds
  - Feature computation intervals
  - Parquet batch size

### 7. Dependencies
- Updated `requirements.txt` with:
  - `evidently==0.4.15` for monitoring
  - `numpy>=1.24.0` for numerical operations
  - `jupyter`, `ipykernel`, `matplotlib`, `seaborn` for EDA

## 📊 Features Computed

1. **Price Features:**
   - `price`: Current trade price
   - `midprice`: (best_bid + best_ask) / 2
   - `return_1s`, `return_5s`, `return_30s`, `return_60s`: Returns over different intervals

2. **Volatility Features:**
   - `volatility`: Rolling standard deviation of returns

3. **Market Microstructure:**
   - `spread_abs`: Absolute bid-ask spread
   - `spread_rel`: Relative bid-ask spread
   - `order_book_imbalance`: (ask - bid) / (ask + bid)

4. **Trading Activity:**
   - `trade_intensity`: Trades per second

5. **Target Variable (computed in EDA):**
   - `future_volatility`: Rolling std of returns over next 60 seconds
   - `label`: Binary label (1 = spike, 0 = normal)

## 🧪 Testing Checklist

- [ ] Run replay script on saved raw data
- [ ] Compare replay features with live consumer features (should be identical)
- [ ] Run EDA notebook to compute threshold
- [ ] Generate Evidently report
- [ ] Verify Evidently report includes drift and data quality metrics

## 📁 File Structure

```
crypto-volatility/
├── features/
│   ├── __init__.py
│   └── featurizer.py          # Feature engineering pipeline
├── scripts/
│   ├── replay.py              # Replay script
│   └── generate_evidently_report.py  # Evidently report generator
├── notebooks/
│   └── eda.ipynb              # EDA notebook
├── docs/
│   └── feature_spec.md        # Feature specification
├── data/
│   └── processed/
│       └── features.parquet   # Generated features
└── reports/
    └── evidently/
        ├── evidently_report.html
        └── evidently_report.json
```

## 🚀 Usage

1. **Generate features from live stream:**
   ```bash
   python features/featurizer.py
   ```

2. **Replay features from raw data:**
   ```bash
   python scripts/replay.py
   ```

3. **Run EDA:**
   ```bash
   jupyter notebook notebooks/eda.ipynb
   ```

4. **Generate Evidently report:**
   ```bash
   python scripts/generate_evidently_report.py
   ```

## 📝 Notes

- Features are computed with a 5-minute lookback window
- Prediction horizon is 60 seconds
- Threshold selection is done via percentile analysis in EDA notebook
- All features are saved to Parquet for efficient storage and analysis

