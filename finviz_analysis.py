"""
Finviz Market Data Analysis
Source: finviz_market_data_20260403_1438.csv
"""

import pandas as pd
import numpy as np

# ──────────────────────────────────────────
# 1. LOAD
# ──────────────────────────────────────────
df = pd.read_csv('finviz_market_data_20260403_1438.csv')
print(f"Shape: {df.shape}")
print(df.dtypes)

# ──────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ──────────────────────────────────────────

# Parse Market Cap string → float (handles T/B/M/K suffixes)
def parse_market_cap(val):
    if val == '-' or pd.isna(val):
        return np.nan
    val = str(val).strip()
    for suffix, multiplier in [('T', 1e12), ('B', 1e9), ('M', 1e6), ('K', 1e3)]:
        if val.endswith(suffix):
            try:
                return float(val[:-1]) * multiplier
            except:
                return np.nan
    try:
        return float(val)
    except:
        return np.nan

df['MarketCap_Num']  = df['Market Cap'].apply(parse_market_cap)
df['PE_Num']         = pd.to_numeric(df['P/E'].replace('-', np.nan), errors='coerce')
df['Change_Num']     = pd.to_numeric(df['Change'].str.replace('%', ''), errors='coerce')
df['Volume_Num']     = pd.to_numeric(df['Volume'].str.replace(',', ''), errors='coerce')
df['Price']          = pd.to_numeric(df['Price'], errors='coerce')

# Cap category from market cap
def cap_category(v):
    if pd.isna(v):   return 'Unknown'
    if v >= 200e9:   return 'Mega Cap'
    if v >= 10e9:    return 'Large Cap'
    if v >= 2e9:     return 'Mid Cap'
    if v >= 300e6:   return 'Small Cap'
    return 'Micro Cap'

df['Cap_Category'] = df['MarketCap_Num'].apply(cap_category)

# Direction flags
df['Is_Gainer']  = df['Change_Num'] > 0
df['Is_Loser']   = df['Change_Num'] < 0
df['Abs_Change'] = df['Change_Num'].abs()

# Log-scaled features (handles skew)
df['Log_Volume']    = np.log1p(df['Volume_Num'])
df['Log_MarketCap'] = np.log1p(df['MarketCap_Num'])

# Price bands
df['Price_Band'] = pd.cut(
    df['Price'].astype(float),
    bins=[0, 5, 20, 50, 100, 200, 500, float('inf')],
    labels=['Penny', 'Very Low', 'Low', 'Mid', 'High', 'Very High', 'Ultra High']
)

# ──────────────────────────────────────────
# 3. BASIC ANALYTICS
# ──────────────────────────────────────────

print("\n=== Market Cap Categories ===")
print(df['Cap_Category'].value_counts())

print("\n=== Gainers / Losers / Flat ===")
print(f"Gainers: {df['Is_Gainer'].sum()}, Losers: {df['Is_Loser'].sum()}, Flat: {(df['Change_Num']==0).sum()}")

print("\n=== Price Bands ===")
print(df['Price_Band'].value_counts().sort_index())

print("\n=== Top 10 Gainers ===")
print(df.nlargest(10, 'Change_Num')[['Ticker','Company','Sector','Price','Change_Num','Cap_Category']].to_string())

print("\n=== Top 10 Losers ===")
print(df.nsmallest(10, 'Change_Num')[['Ticker','Company','Sector','Price','Change_Num','Cap_Category']].to_string())

print("\n=== Sector Stats ===")
sector_stats = df.groupby('Sector').agg(
    Count        = ('Ticker', 'count'),
    Avg_Change   = ('Change_Num', 'mean'),
    Gainers      = ('Is_Gainer', 'sum'),
    Losers       = ('Is_Loser', 'sum'),
    Avg_Price    = ('Price', 'mean'),
    Avg_PE       = ('PE_Num', 'mean'),
    Total_Volume = ('Volume_Num', 'sum')
).round(2)
sector_stats['Win_Rate_pct'] = (sector_stats['Gainers'] / sector_stats['Count'] * 100).round(1)
print(sector_stats.sort_values('Avg_Change', ascending=False).to_string())

print("\n=== Cap Category vs Avg Change ===")
print(df.groupby('Cap_Category')['Change_Num'].agg(['mean', 'std', 'count']).round(3))

print("\n=== Correlation Matrix ===")
numeric_cols = ['Price', 'Change_Num', 'Volume_Num', 'PE_Num', 'MarketCap_Num']
print(df[numeric_cols].corr().round(3))

print("\n=== Country Stats ===")
country_stats = df.groupby('Country').agg(
    Count      = ('Ticker', 'count'),
    Avg_Change = ('Change_Num', 'mean'),
    Gainers    = ('Is_Gainer', 'sum'),
).round(2)
country_stats['Win_Rate'] = (country_stats['Gainers'] / country_stats['Count'] * 100).round(1)
print(country_stats.sort_values('Count', ascending=False).head(15))

# ──────────────────────────────────────────
# 4. OUTLIER DETECTION
# ──────────────────────────────────────────

# Z-score outliers on daily % change (|z| > 3)
mean_c = df['Change_Num'].mean()
std_c  = df['Change_Num'].std()
df['Change_ZScore'] = (df['Change_Num'] - mean_c) / std_c

print("\n=== Z-Score Outliers (|z| > 3) ===")
outliers = df[df['Change_ZScore'].abs() > 3][[
    'Ticker', 'Company', 'Sector', 'Change_Num', 'Change_ZScore', 'Cap_Category', 'Volume_Num'
]]
print(outliers.sort_values('Change_ZScore', ascending=False).to_string())

# IQR outliers on P/E
q1 = df['PE_Num'].quantile(0.25)
q3 = df['PE_Num'].quantile(0.75)
iqr = q3 - q1
upper_fence = q3 + 1.5 * iqr
pe_outliers = df[df['PE_Num'] > upper_fence][['Ticker', 'Company', 'Sector', 'PE_Num', 'Price', 'Cap_Category']]
print(f"\n=== P/E IQR Outliers (>{upper_fence:.1f}) — {len(pe_outliers)} tickers ===")
print(pe_outliers.nlargest(10, 'PE_Num').to_string())

# ──────────────────────────────────────────
# 5. INDUSTRY DRILL-DOWN
# ──────────────────────────────────────────

ind = df.groupby('Industry').agg(
    Count      = ('Ticker', 'count'),
    Avg_Change = ('Change_Num', 'mean'),
    Win_Rate   = ('Is_Gainer', 'mean'),
    Avg_Volume = ('Volume_Num', 'mean')
).round(3)
ind['Win_Rate_pct'] = (ind['Win_Rate'] * 100).round(1)
ind_filtered = ind[ind['Count'] >= 5].sort_values('Avg_Change', ascending=False)

print("\n=== Top 20 Industries by Avg Change (min 5 tickers) ===")
print(ind_filtered.head(20).to_string())

print("\n=== Bottom 10 Industries ===")
print(ind_filtered.tail(10).to_string())

# ──────────────────────────────────────────
# 6. P/E ANALYSIS
# ──────────────────────────────────────────

print("\n=== Median P/E by Sector ===")
pe_sector = df[df['PE_Num'].notna()].groupby('Sector')['PE_Num'].agg(['mean', 'median', 'count']).round(2)
print(pe_sector.sort_values('median', ascending=False))

print("\n=== Losing Stocks with High P/E (PE > 50) — potential value traps ===")
risky = df[(df['PE_Num'] > 50) & df['Is_Loser']][[
    'Ticker', 'Company', 'Sector', 'PE_Num', 'Change_Num', 'Price', 'Cap_Category'
]]
print(risky.sort_values('PE_Num', ascending=False).head(15).to_string())

# ──────────────────────────────────────────
# 7. VOLUME ANALYSIS
# ──────────────────────────────────────────

print("\n=== Top 10 Tickers by Volume ===")
print(df.nlargest(10, 'Volume_Num')[['Ticker', 'Company', 'Sector', 'Volume_Num', 'Change_Num', 'Cap_Category']].to_string())

print("\n=== Avg Volume by Sector ===")
print(df.groupby('Sector')['Volume_Num'].mean().sort_values(ascending=False).apply(lambda x: f"{x/1e6:.2f}M"))

# ──────────────────────────────────────────
# 8. PRICE SEGMENT ANALYSIS
# ──────────────────────────────────────────

print("\n=== Penny Stocks (<$5) ===")
penny = df[df['Price'] < 5]
print(f"Count: {len(penny)}, Gainers: {penny['Is_Gainer'].sum()}, Losers: {penny['Is_Loser'].sum()}")
print(f"Avg Change: {penny['Change_Num'].mean():.2f}%, Std Dev: {penny['Change_Num'].std():.2f}%")

print("\n=== High-Priced Stocks (>$100) ===")
highp = df[df['Price'] > 100]
print(f"Count: {len(highp)}, Gainers: {highp['Is_Gainer'].sum()}, Losers: {highp['Is_Loser'].sum()}")
print(f"Avg Change: {highp['Change_Num'].mean():.2f}%, Std Dev: {highp['Change_Num'].std():.2f}%")

# ──────────────────────────────────────────
# 9. EXPORT ENRICHED CSV
# ──────────────────────────────────────────

df.to_csv('finviz_enriched.csv', index=False)
print("\nEnriched CSV saved to finviz_enriched.csv")
print(f"New columns added: {[c for c in df.columns if c not in ['No','Ticker','Company','Sector','Industry','Country','Market Cap','P/E','Price','Change','Volume']]}")
