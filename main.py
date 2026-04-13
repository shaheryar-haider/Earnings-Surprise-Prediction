"""
FINE 695 – Individual Assignment
Strategy: Earnings Surprise Prediction (Long-Short)
- Predict: eps_actual - eps_meanest (earnings surprise)
- Long:  Top 50 stocks with highest predicted positive surprise
- Short: Bottom 50 stocks with most negative predicted surprise
- Models: Lasso, Ridge, Elastic Net (CPU) + NN2 (GPU)
- Benchmark: S&P 500 (sp_ret from mkt_ind.csv)
Author: Shaheryar Haider
"""

# ── 0. Imports ────────────────────────────────────────────────────────────────
import datetime
import warnings
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

warnings.filterwarnings("ignore")
pd.set_option("mode.chained_assignment", None)

# ── 1. Paths ──────────────────────────────────────────────────────────────────
work_dir  = r"C:/Users/Shariq/Documents/Individual Assignment data & codes/"
data_file = work_dir + "mma_sample_v2.csv"
char_file = work_dir + "factor_char_list.csv"
mkt_file  = work_dir + "mkt_ind.csv"
out_dir   = work_dir + "output/"
os.makedirs(out_dir, exist_ok=True)

# ── 2. PyTorch / GPU Setup ────────────────────────────────────────────────────
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── 3. NN2 Architecture ───────────────────────────────────────────────────────
class NN2(nn.Module):
    def __init__(self, input_dim):
        super(NN2, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return torch.squeeze(self.fc3(x))


def _run_epoch(model, loader, optimizer, criterion, training):
    model.train() if training else model.eval()
    total = 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            loss = criterion(model(X_b), y_b)
            if training:
                loss += 0.0001 * torch.norm(model.fc1.weight, p=1)
                loss.backward()
                optimizer.step()
        total += loss.item() * len(X_b)
    return total / len(loader.dataset)


def train_nn2(X_tr, Y_tr, X_va, Y_va, X_te,
              batch_size=10000, max_epochs=100, patience=5):
    """Train NN2 on GPU and return OOS predictions."""
    tr_ld = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(Y_tr)),
        batch_size=batch_size, shuffle=True)
    va_ld = DataLoader(
        TensorDataset(torch.FloatTensor(X_va), torch.FloatTensor(Y_va)),
        batch_size=batch_size)

    model     = NN2(X_tr.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    best_val, wait, best_state = np.inf, 0, None
    for _ in range(max_epochs):
        _run_epoch(model, tr_ld, optimizer, criterion, training=True)
        val_loss = _run_epoch(model, va_ld, None, criterion, training=False)
        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
        scheduler.step()

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_te).to(device)).cpu().numpy()
    return preds

# ── 4. Load Data ──────────────────────────────────────────────────────────────
print(f"\n[{datetime.datetime.now():%H:%M:%S}] Loading data...")
raw = pd.read_csv(data_file, parse_dates=["date"], low_memory=False)
mkt = pd.read_csv(mkt_file)

char_list = list(pd.read_csv(char_file)["variable"].values)
char_list = [c for c in char_list if c in raw.columns]
print(f"  Using {len(char_list)} characteristics as predictors")

# ── 5. Construct Target: Earnings Surprise ────────────────────────────────────
print(f"[{datetime.datetime.now():%H:%M:%S}] Constructing earnings surprise...")
raw["earnings_surprise"] = raw["eps_actual"] - raw["eps_meanest"]
raw_surprise = raw[raw["earnings_surprise"].notna()].copy()
print(f"  Rows with earnings surprise: {len(raw_surprise):,} "
      f"({len(raw_surprise)/len(raw)*100:.1f}% of dataset)")

# ── 6. Cross-sectional Rank Transform to [-1, 1] ─────────────────────────────
print(f"[{datetime.datetime.now():%H:%M:%S}] Rank-transforming characteristics...")
binary_vars = ["convind", "divi", "divo", "rd", "securedind", "sin"]
binary_vars = [v for v in binary_vars if v in char_list]

adj_list = []
for date, grp in raw_surprise.groupby("date"):
    grp = grp.copy()
    for var in char_list:
        med = grp[var].median(skipna=True)
        grp[var] = grp[var].fillna(med)
        if var not in binary_vars:
            grp[var] = grp[var].rank(method="dense") - 1
            vmax = grp[var].max()
            grp[var] = (grp[var] / vmax) * 2 - 1 if vmax > 0 else 0
    adj_list.append(grp)

data = pd.concat(adj_list, ignore_index=True)
print(f"  Dataset shape: {data.shape}")

# ── 7. Expanding-Window Estimation ───────────────────────────────────────────
ret_var  = "earnings_surprise"
starting = pd.Timestamp("2000-01-01")
counter  = 0
pred_out = []

print(f"\n[{datetime.datetime.now():%H:%M:%S}] Starting expanding-window estimation...")

while True:
    cutoffs = [
        starting,
        starting + pd.DateOffset(years=8  + counter),
        starting + pd.DateOffset(years=10 + counter),
        starting + pd.DateOffset(years=11 + counter),
    ]
    if cutoffs[3] > pd.Timestamp("2024-01-01"):
        break

    train    = data[(data["date"] >= cutoffs[0]) & (data["date"] < cutoffs[1])].copy()
    validate = data[(data["date"] >= cutoffs[1]) & (data["date"] < cutoffs[2])].copy()
    test     = data[(data["date"] >= cutoffs[2]) & (data["date"] < cutoffs[3])].copy()

    if len(test) == 0:
        break

    print(f"  Window {counter}: train {cutoffs[0].year}–{cutoffs[1].year-1} | "
          f"val {cutoffs[1].year}–{cutoffs[2].year-1} | "
          f"test {cutoffs[2].year}  (n_test={len(test)})")

    scaler     = StandardScaler().fit(train[char_list])
    X_train    = scaler.transform(train[char_list])
    X_val      = scaler.transform(validate[char_list])
    X_test     = scaler.transform(test[char_list])
    Y_train    = train[ret_var].values
    Y_val      = validate[ret_var].values
    Y_mean     = Y_train.mean()
    Y_train_dm = Y_train - Y_mean
    Y_val_dm   = Y_val   - Y_mean

    reg_pred = test[["year", "month", "date", "permno",
                      "stock_ticker", "comp_name",
                      "stock_exret", ret_var]].copy()

    # ── Lasso ─────────────────────────────────────────────────────────────
    lambdas = np.arange(-4, 4.1, 0.5)
    mses = [mean_squared_error(Y_val,
            Lasso(alpha=10**l, max_iter=1000000, fit_intercept=False)
            .fit(X_train, Y_train_dm).predict(X_val) + Y_mean)
            for l in lambdas]
    best_l = lambdas[np.argmin(mses)]
    m = Lasso(alpha=10**best_l, max_iter=1000000,
              fit_intercept=False).fit(X_train, Y_train_dm)
    reg_pred["lasso"] = m.predict(X_test) + Y_mean

    # ── Ridge ─────────────────────────────────────────────────────────────
    lambdas = np.arange(-1, 8.1, 0.5)
    mses = [mean_squared_error(Y_val,
            Ridge(alpha=(10**l)*0.5, fit_intercept=False)
            .fit(X_train, Y_train_dm).predict(X_val) + Y_mean)
            for l in lambdas]
    best_l = lambdas[np.argmin(mses)]
    m = Ridge(alpha=(10**best_l)*0.5,
              fit_intercept=False).fit(X_train, Y_train_dm)
    reg_pred["ridge"] = m.predict(X_test) + Y_mean

    # ── Elastic Net ───────────────────────────────────────────────────────
    lambdas = np.arange(-4, 4.1, 0.5)
    mses = [mean_squared_error(Y_val,
            ElasticNet(alpha=10**l, max_iter=1000000, fit_intercept=False)
            .fit(X_train, Y_train_dm).predict(X_val) + Y_mean)
            for l in lambdas]
    best_l = lambdas[np.argmin(mses)]
    m = ElasticNet(alpha=10**best_l, max_iter=1000000,
                   fit_intercept=False).fit(X_train, Y_train_dm)
    reg_pred["en"] = m.predict(X_test) + Y_mean

    # ── NN2 (GPU) ─────────────────────────────────────────────────────────
    reg_pred["nn2"] = train_nn2(
        X_train, Y_train_dm, X_val, Y_val_dm, X_test) + Y_mean

    pred_out.append(reg_pred)
    counter += 1

pred_out = pd.concat(pred_out, ignore_index=True)
pred_out.to_csv(out_dir + "predictions.csv", index=False)
print(f"\n[{datetime.datetime.now():%H:%M:%S}] All predictions saved.")

# ── 8. OOS R² ─────────────────────────────────────────────────────────────────
print("\n=== Out-of-Sample R² (×100) ===")
y_real = pred_out[ret_var].values
oos_r2 = {}
for mdl in ["lasso", "ridge", "en", "nn2"]:
    y_hat = pred_out[mdl].values
    r2 = 1 - np.sum((y_real - y_hat)**2) / np.sum(y_real**2)
    oos_r2[mdl] = r2
    print(f"  {mdl:6s}: {r2*100:.4f}")

best_model = max(oos_r2, key=oos_r2.get)
print(f"\nBest model by OOS R²: {best_model.upper()}")

# ── 9. Portfolio Construction ─────────────────────────────────────────────────
print(f"\n[{datetime.datetime.now():%H:%M:%S}] Building long-short portfolio...")
N_STOCKS = 50

port_rows      = []
top10_holdings = []

for (yr, mo), grp in pred_out.groupby(["year", "month"]):
    grp = grp.dropna(subset=[best_model, "stock_exret"])
    if len(grp) < N_STOCKS * 2:
        continue

    grp_sorted = grp.sort_values(best_model)
    short_leg  = grp_sorted.head(N_STOCKS)
    long_leg   = grp_sorted.tail(N_STOCKS)

    long_ret  = long_leg["stock_exret"].mean()
    short_ret = short_leg["stock_exret"].mean()
    ls_ret    = long_ret - short_ret

    port_rows.append({
        "year": yr, "month": mo,
        "long_ret": long_ret,
        "short_ret": short_ret,
        "ls_ret": ls_ret,
    })

    for _, row in long_leg.iterrows():
        top10_holdings.append({
            "year": yr, "month": mo,
            "permno": row["permno"],
            "ticker": row.get("stock_ticker", ""),
            "company": row.get("comp_name", ""),
            "predicted_surprise": row[best_model],
        })

port = pd.DataFrame(port_rows)
port = port.merge(mkt, on=["year", "month"], how="inner")
port = port.sort_values(["year", "month"]).reset_index(drop=True)
print(f"  Portfolio months: {len(port)}")

# ── 10. Performance Metrics ───────────────────────────────────────────────────
def perf_stats(rets):
    ann_ret = rets.mean() * 12
    ann_std = rets.std()  * np.sqrt(12)
    sharpe  = ann_ret / ann_std
    cum     = np.log(1 + rets).cumsum()
    max_dd  = (cum.cummax() - cum).max()
    max_1m  = rets.min()
    return ann_ret, ann_std, sharpe, max_dd, max_1m

a_ret, a_std, a_sr, a_dd, a_1m = perf_stats(port["ls_ret"])
s_ret, s_std, s_sr, s_dd, s_1m = perf_stats(port["sp_ret"])

port["mkt_rf"] = port["sp_ret"] - port["rf"]
reg = smf.ols("ls_ret ~ mkt_rf", data=port).fit(
      cov_type="HAC", cov_kwds={"maxlags": 3}, use_t=True)
alpha_ann  = reg.params["Intercept"] * 12
beta       = reg.params["mkt_rf"]
t_stat     = reg.tvalues["Intercept"]
info_ratio = reg.params["Intercept"] / np.sqrt(reg.mse_resid) * np.sqrt(12)

def compute_turnover(df, signal, n=50):
    monthly_longs = {}
    for (yr, mo), grp in df.groupby(["year", "month"]):
        grp = grp.dropna(subset=[signal])
        if len(grp) < n * 2:
            continue
        monthly_longs[(yr, mo)] = set(grp.nlargest(n, signal)["permno"].values)
    dates = sorted(monthly_longs.keys())
    turnovers = []
    for i in range(1, len(dates)):
        prev = monthly_longs[dates[i-1]]
        curr = monthly_longs[dates[i]]
        if curr:
            turnovers.append(len(curr - prev) / len(curr))
    return np.mean(turnovers) if turnovers else 0

turnover = compute_turnover(pred_out, best_model)

print(f"\n{'Metric':<28} {'Active (L/S)':>14} {'SPY':>10}")
print("-" * 55)
for label, av, sv in [
    ("Ann. Avg Return",      a_ret,      s_ret),
    ("Ann. Std Dev",         a_std,      s_std),
    ("Sharpe Ratio (ann.)",  a_sr,       s_sr),
    ("Alpha (ann., CAPM)",   alpha_ann,  0.0),
    ("Alpha t-stat",         t_stat,     0.0),
    ("Beta",                 beta,       1.0),
    ("Information Ratio",    info_ratio, 0.0),
    ("Max Drawdown",         a_dd,       s_dd),
    ("Max 1-Month Loss",     a_1m,       s_1m),
    ("Turnover (monthly)",   turnover,   0.0),
]:
    print(f"  {label:<26} {av:>14.4f} {sv:>10.4f}")

# ── 11. Top 10 Holdings ───────────────────────────────────────────────────────
holdings_df = pd.DataFrame(top10_holdings)
top10 = (holdings_df.groupby(["permno", "ticker", "company"])
         .size().reset_index(name="months_in_long")
         .sort_values("months_in_long", ascending=False).head(10))
print("\n=== Top 10 Holdings (by frequency in long portfolio) ===")
print(top10.to_string(index=False))
top10.to_csv(out_dir + "top10_holdings.csv", index=False)

# ── 12. Save Metrics ──────────────────────────────────────────────────────────
metrics_df = pd.DataFrame([{
    "Strategy":          f"Active L/S ({best_model.upper()})",
    "OOS R2 (x100)":     oos_r2[best_model] * 100,
    "Ann. Return":       a_ret,
    "Ann. Std Dev":      a_std,
    "Sharpe Ratio":      a_sr,
    "Alpha (ann.)":      alpha_ann,
    "Alpha t-stat":      t_stat,
    "Beta":              beta,
    "Information Ratio": info_ratio,
    "Max Drawdown":      a_dd,
    "Max 1-Month Loss":  a_1m,
    "Turnover":          turnover,
}, {
    "Strategy":          "SPY (Passive)",
    "OOS R2 (x100)":     None,
    "Ann. Return":       s_ret,
    "Ann. Std Dev":      s_std,
    "Sharpe Ratio":      s_sr,
    "Alpha (ann.)":      0.0,
    "Alpha t-stat":      0.0,
    "Beta":              1.0,
    "Information Ratio": 0.0,
    "Max Drawdown":      s_dd,
    "Max 1-Month Loss":  s_1m,
    "Turnover":          0.0,
}])
metrics_df.to_csv(out_dir + "portfolio_metrics.csv", index=False)
print(f"\nMetrics saved → {out_dir}portfolio_metrics.csv")

# ── 13. Cumulative Return Plot ────────────────────────────────────────────────
port["cum_ls"]  = (1 + port["ls_ret"]).cumprod()
port["cum_spy"] = (1 + port["sp_ret"]).cumprod()
port["period"]  = pd.to_datetime(port[["year", "month"]].assign(day=1))

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(port["period"], port["cum_ls"],
        label=f"Active L/S ({best_model.upper()})",
        color="steelblue", linewidth=2)
ax.plot(port["period"], port["cum_spy"],
        label="SPY (Buy & Hold)",
        color="darkorange", linewidth=2, linestyle="--")
ax.set_title("Cumulative OOS Returns: Earnings Surprise Strategy vs S&P 500",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Growth of $1")
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir + "cumulative_returns.png", dpi=150)
plt.show()
print(f"Chart saved → {out_dir}cumulative_returns.png")

print(f"\n[{datetime.datetime.now():%H:%M:%S}] Done!")