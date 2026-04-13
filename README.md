## Earnings Surprise Prediction Strategy

**Author:** Shaheryar Haider  

---

## Strategy Overview

This project implements a **long-short equity strategy** based on predicting earnings surprises using machine learning. The core idea is that stocks with predictable positive earnings surprises tend to outperform, while stocks likely to disappoint tend to underperform — a well-documented phenomenon known as **Post-Earnings Announcement Drift (PEAD)**.

**Target Variable:** Earnings Surprise = `eps_actual − eps_meanest` (actual EPS minus mean analyst consensus estimate)

**Portfolio:**
- **Long:** Top 50 stocks with highest predicted positive surprise
- **Short:** Bottom 50 stocks with most negative predicted surprise
- **Rebalancing:** Monthly
- **Weighting:** Equal-weighted

---

## Models Used

| Model | Type | Device |
|---|---|---|
| Lasso | Penalized Linear | CPU |
| Ridge | Penalized Linear | CPU |
| Elastic Net | Penalized Linear | CPU |
| NN2 | 2-Layer Neural Network (32→16→1) | GPU |

The best model is selected automatically based on out-of-sample R².

---

## Training Setup

Expanding window approach following Gu et al. (2020):

| Split | Period |
|---|---|
| Initial Training | 2000–2007 (8 years, expanding) |
| Validation | Next 2 years (hyperparameter tuning) |
| OOS Test | Following year |
| Total OOS Period | 2010–2023 (14 windows) |

---

## Out-of-Sample Results

| Model | OOS R² (×100) |
|---|---|
| Lasso | -0.2935 |
| Ridge | -0.2968 |
| Elastic Net | -0.2935 |
| **NN2 (selected)** | **-0.2291** |

---

## Portfolio Performance (OOS 2010–2023)

| Metric | Active L/S (NN2) | SPY (Passive) |
|---|---|---|
| Ann. Avg Return | 2.44% | 11.52% |
| Ann. Std Dev | 10.70% | 14.82% |
| Sharpe Ratio | 0.23 | 0.78 |
| Alpha (ann., CAPM) | 3.76% | — |
| Alpha t-stat | 1.20 | — |
| Beta | -0.12 | 1.00 |
| Information Ratio | 0.36 | — |
| Max Drawdown | 0.26 | 0.28 |
| Max 1-Month Loss | -9.57% | -12.51% |
| Monthly Turnover | 51.3% | — |

---

## Repository Structure

```
fine695_individual/
│
├── individual_assignment.py    # Main run file
├── README.md                   # This file
├── .gitignore                  # Excludes large data files
│
└── output/
    ├── predictions.csv         # OOS predictions for all models
    ├── portfolio_metrics.csv   # Performance statistics
    ├── top10_holdings.csv      # Top 10 holdings by frequency
    └── cumulative_returns.png  # Cumulative return chart vs SPY
```

---

## How to Run

**1. Install dependencies:**
```bash
pip install pandas numpy scikit-learn torch statsmodels matplotlib
```

**2. Place data files in the working directory:**
- `mma_sample_v2.csv`
- `factor_char_list.csv`
- `mkt_ind.csv`

**3. Update the working directory path in the script:**
```python
work_dir = r"your/path/here/"
```

**4. Run:**
```bash
python individual_assignment.py
```

**Expected runtime:** ~5-6 hours on CPU, ~1-2 hours with GPU

---

## Data

- **Main dataset:** `mma_sample_v2.csv` — large-cap US stocks, Jan 2000–Dec 2023
- **Characteristics:** 147 firm-specific signals from `factor_char_list.csv`
- **Market data:** `mkt_ind.csv` — monthly S&P 500 returns and risk-free rate
- **Note:** Large data files are excluded from this repo via `.gitignore`

---

## Key References

- Gu, S., Kelly, B., and Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223–2273.
- Goyenko, R. and Zhang, C. (2022). The Joint Cross Section of Options and Stock Returns Predictability with Big Data and Machine Learning. McGill University working paper.
- Chapados, N., Fan, Z., Goyenko, R., et al. (2023). Can AI Read the Minds of Corporate Executives? McGill University working paper.
