import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from forecast import METHODS

panel = pd.read_csv("data/panel_top300.csv", index_col=0)
Y = panel.values.astype(float); n_prod, T = Y.shape
FIRST = 16  # first forecast origin: 2022Q1, 16 quarters of history
orig = list(range(FIRST, T))

# ---- backtest: one-step-ahead forecasts with expanding history
F = {m: np.zeros((n_prod, len(orig))) for m in METHODS}
for i in range(n_prod):
    for k, t in enumerate(orig):
        for m, fn in METHODS.items():
            F[m][i, k] = fn(Y[i, :t])
A = Y[:, FIRST:]

rows = []
for m in METHODS:
    e = F[m] - A
    scale = np.array([np.mean(np.abs(np.diff(Y[i, :t]))) for i in range(n_prod) for t in [FIRST]])
    mase = np.abs(e).mean(1) / scale
    rows.append(dict(method=m, median_MASE=np.median(mase), mean_MASE=mase.mean(),
                     WAPE=np.abs(e).sum() / A.sum(), bias_pct=100 * e.sum() / A.sum(),
                     median_abs_pct_err=100 * np.median(np.abs(e / A))))
res = pd.DataFrame(rows).round(3); res.to_csv("data/forecast_accuracy.csv", index=False); print(res.to_string())

# ---- segmentation: volume (ABC) x variability (XYZ) from history before the test period
hist = Y[:, :FIRST]
vol = hist.sum(1); order = np.argsort(-vol); cum = np.cumsum(vol[order]) / vol.sum()
abc = np.empty(n_prod, dtype="<U1"); abc[order[cum <= 0.7]] = "A"; abc[order[(cum > 0.7) & (cum <= 0.9)]] = "B"; abc[order[cum > 0.9]] = "C"
# variability after removing a log-linear trend and quarterly pattern (so seasonality is not counted as noise)
cv = np.empty(n_prod)
for i in range(n_prod):
    ly = np.log(hist[i]); X = np.c_[np.ones(FIRST), np.arange(FIRST), np.eye(4)[np.arange(FIRST) % 4][:, 1:]]
    b, *_ = np.linalg.lstsq(X, ly, rcond=None); cv[i] = np.std(ly - X @ b)
q1, q2 = np.quantile(cv, [1 / 3, 2 / 3])
xyz = np.where(cv <= q1, "X", np.where(cv <= q2, "Y", "Z"))
seg = pd.DataFrame({"product": panel.index, "abc": abc, "xyz": xyz, "residual_sd_log": cv.round(3)})
seg.to_csv("data/segments.csv", index=False)
print(pd.crosstab(seg.abc, seg.xyz))

segrows = []
for s in ["X", "Y", "Z"]:
    mk = xyz == s
    for m in METHODS:
        e = np.abs(F[m][mk] - A[mk]).sum() / A[mk].sum()
        segrows.append(dict(variability=s, method=m, WAPE=round(e, 3)))
sg = pd.DataFrame(segrows).pivot(index="method", columns="variability", values="WAPE"); sg.to_csv("data/wape_by_variability.csv"); print(sg)
best_by_seg = sg.idxmin(); print("best by segment", best_by_seg.to_dict())
# choose a method per variability class using the FIRST HALF of the test period, score on the second half (no peeking)
h = len(orig) // 2
pick = {}
for s_ in "XYZ":
    mk = xyz == s_
    pick[s_] = min(METHODS, key=lambda m: np.abs(F[m][mk][:, :h] - A[mk][:, :h]).sum())
err = sum(np.abs(F[pick[s_]][xyz == s_][:, h:] - A[xyz == s_][:, h:]).sum() for s_ in "XYZ") / A[:, h:].sum()
print("picked", pick, "second-half WAPE", round(err, 4), "| naive", round(np.abs(F["naive"][:, h:] - A[:, h:]).sum() / A[:, h:].sum(), 4),
      "| HW", round(np.abs(F["Holt-Winters (log, damped)"][:, h:] - A[:, h:]).sum() / A[:, h:].sum(), 4))
pd.DataFrame([dict(variability=k, picked_method=v) for k, v in pick.items()]).assign(second_half_WAPE_selected=round(err, 4)).to_csv("data/segment_selection.csv", index=False)

# ---- inventory simulation: order-up-to level = forecast + z * rolling error sd (relative), lost sales, leftover carries over
from statistics import NormalDist
ND = NormalDist()

def simulate(Fm, z):
    served = 0.0; demand = 0.0; onhand_end = 0.0
    for i in range(n_prod):
        stock = 0.0
        rel = []  # past relative forecast errors
        for k in range(len(orig)):
            f = Fm[i, k]
            sd = np.std(rel[-8:]) if len(rel) >= 3 else 0.15
            target = f * (1 + z * sd)
            # stock carried in counts towards the target; order brings the position up to target
            stock = max(stock, target)
            d = A[i, k]
            sell = min(stock, d)
            served += sell; demand += d
            stock -= sell; onhand_end += stock
            rel.append((d - f) / f)
    return served / demand, onhand_end / demand  # fill rate, leftover stock per unit of demand

curve = []
for m in METHODS:
    for sl in [0.80, 0.90, 0.95, 0.975, 0.99]:
        z = ND.inv_cdf(sl)
        fr, inv = simulate(F[m], z)
        curve.append(dict(method=m, z_target=sl, fill_rate=round(fr, 4), leftover_stock_per_demand=round(inv, 4)))
cv_df = pd.DataFrame(curve); cv_df.to_csv("data/service_vs_inventory.csv", index=False)
print(cv_df.pivot(index="method", columns="z_target", values="fill_rate"))
print(cv_df.pivot(index="method", columns="z_target", values="leftover_stock_per_demand"))

# ---- the 2020 shock and one example product
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
ms = res.sort_values("WAPE")
ax[0].barh(ms.method[::-1], ms.WAPE[::-1] * 100, color="#4d4d4d"); ax[0].set_xlabel("WAPE %, 2022-2025 one-quarter-ahead"); ax[0].set_title("Forecast error by method (300 drugs)")
for m in METHODS:
    c = cv_df[cv_df.method == m]; ax[1].plot(c.leftover_stock_per_demand, c.fill_rate * 100, "o-", label=m, ms=4)
ax[1].set_xlabel("Leftover stock per unit of demand"); ax[1].set_ylabel("Fill rate %"); ax[1].set_title("Service vs inventory trade-off"); ax[1].legend(fontsize=7)
tot = Y.sum(0) / 1e9; ax[2].plot(range(T), tot, color="#b2182b"); ax[2].set_xticks(range(0, T, 4)); ax[2].set_xticklabels([f"{2018 + t // 4}" for t in range(0, T, 4)])
ax[2].set_ylabel("Billion units"); ax[2].set_title("Total Medicaid units, 300 drugs"); ax[2].axvline(FIRST - .5, ls="--", color="k", lw=1)
plt.tight_layout(); plt.savefig("figures/forecast_and_inventory.png", dpi=100)
