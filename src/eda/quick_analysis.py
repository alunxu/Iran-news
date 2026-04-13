"""Quick analysis of newly collected 2020-2026 data for report adequacy."""
import pandas as pd

df = pd.read_parquet("data/iran_articles.parquet")
df["year"] = pd.to_datetime(df["pub_date"]).dt.year
text = (df["headline"].fillna("") + " " + df["abstract"].fillna("")).str.lower()

# Volume spike analysis
print("=== YEARLY VOLUME 2020-2026 ===")
for yr in range(2020, 2027):
    n = (df["year"] == yr).sum()
    if n: print(f"  {yr}: {n}")

# Key keyword trends
print("\n=== KEY TRENDS 2020-2026 ===")
for kw in ["nuclear", "missile", "israel", "protest", "women", "diplomacy"]:
    mask = text.str.contains(kw, na=False)
    for yr in [2022, 2023, 2024, 2025]:
        yr_m = df["year"] == yr
        t, h = yr_m.sum(), (yr_m & mask).sum()
        if t and h / t > 0.02:
            print(f"  {kw} {yr}: {h}/{t} = {h/t*100:.1f}%")

# Diplomacy gap across all periods
print("\n=== DIPLOMACY GAP ===")
diplo = text.str.contains("diplom", na=False)
for lbl, s, e in [("Hostage",1979,1981),("JCPOA",2015,2017),("WLF",2022,2023),("Direct",2024,2026)]:
    m = (df["year"] >= s) & (df["year"] <= e)
    t, d = m.sum(), (m & diplo).sum()
    if t: print(f"  {lbl}: {d/t*100:.1f}%")

# Re-narration co-occurrences
print("\n=== RE-NARRATION ===")
for a, b in [("revolution","hostage"),("revolution","nuclear"),("nuclear","israel"),("missile","israel")]:
    print(f"  {a}+{b}: {(text.str.contains(a) & text.str.contains(b)).sum()}")

print("\nTotal corpus:", len(df))
