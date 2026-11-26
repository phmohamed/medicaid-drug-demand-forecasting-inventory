"""Combine the yearly files into one national quarterly panel and pick the products to model."""
import glob, pandas as pd

d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("data/units_20*.csv"))])
d["product"] = d.product_name.str.upper().str.strip()
d["t"] = (d.year - 2018) * 4 + d.quarter - 1
p = d.pivot_table(index="product", columns="t", values="units", aggfunc="sum")
full = p[p.notna().all(axis=1) & (p > 0).all(axis=1)]
total = full.sum(axis=1).sort_values(ascending=False)
keep = total.head(300).index
panel = full.loc[keep]
panel.columns = [f"{2018 + t // 4}Q{t % 4 + 1}" for t in panel.columns]
panel.to_csv("data/panel_top300.csv")
print("products with any record", p.shape[0], "with all 32 quarters positive", len(full), "kept", len(panel))
print("share of all units in kept products", round(panel.sum().sum() / d.units.sum(), 3))
