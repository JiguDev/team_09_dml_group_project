# notebooks/eda.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("data/raw/city_day.csv")
print(df.shape)
print(df.columns)
# show AQI distribution
for c in df.columns:
    if c.lower() == 'aqi' or 'aqi' in c.lower():
        df[c] = pd.to_numeric(df[c], errors='coerce')
        sns.histplot(df[c].dropna(), bins=60)
        plt.title("AQI distribution")
        plt.savefig("reports/fig_aqi_dist.png")
        break
# per city heatmap of PM2.5 medians (if PM2.5 present)
if any('pm2' in col.lower() for col in df.columns):
    pm_col = [col for col in df.columns if 'pm2' in col.lower()][0]
    city_col = next((col for col in df.columns if col.lower() == 'city' or 'station' in col.lower()), None)
    if city_col:
        agg = df.groupby(city_col)[pm_col].median().sort_values(ascending=False).head(30)
        sns.barplot(x=agg.values, y=agg.index)
        plt.xlabel(pm_col)
        plt.title("Top 30 cities by median PM2.5")
        plt.tight_layout()
        plt.savefig("reports/fig_top_cities_pm25.png")
print("EDA images saved to reports/")
