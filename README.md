import pandas as pd
import glob

# Find the Excel file automatically (name is long, this avoids typos)
excel_file = glob.glob("tmdb_movies_2024_*.xlsx")[0]
print(f"Loading: {excel_file}")

df = pd.read_excel(excel_file)

print(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns")
df.head()
df.info()
df.columns
df["ROI"] = (df["revenue"] - df["budget"]) / df["budget"]
df[["revenue", "budget", "ROI"]].head()
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

if "ROI" not in df.columns:
    df["ROI"] = np.where(
        df["budget"].ne(0),
        (df["revenue"] - df["budget"]) / df["budget"],
        np.nan,
    )

country_roi = (
    df.assign(prod_country=df["production_countries"].fillna("").str.split(","))
    .explode("prod_country")
    .assign(prod_country=lambda x: x["prod_country"].str.strip())
    .loc[lambda x: x["prod_country"].ne("")]
    [["prod_country", "ROI"]]
)

avg_roi_by_country = (
    country_roi.groupby("prod_country")["ROI"]
    .mean()
    .dropna()
    .sort_values(ascending=False)
)

top_countries = avg_roi_by_country.head(15)

fig, ax = plt.subplots(figsize=(12, 6))
top_countries.plot(kind="bar", ax=ax, color="steelblue")
ax.set_title("Average ROI by Production Country")
ax.set_xlabel("Production Country")
ax.set_ylabel("Average ROI")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()

top_countries
%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt

# Explode the comma-separated genres into one row per genre
genre_series = (
    df["genres"]
    .fillna("")
    .astype(str)
    .str.split(",")
    .explode()
    .str.strip()
    .loc[lambda s: s != ""]
)

# Count each genre and convert to percentages
genre_counts = genre_series.value_counts()
genre_pct = genre_counts / genre_counts.sum() * 100

# Show max and min genres
max_genre = genre_pct.idxmax()
min_genre = genre_pct.idxmin()
max_pct = genre_pct.max()
min_pct = genre_pct.min()

print(f"Highest percentage genre: {max_genre} ({max_pct:.2f}%)")
print(f"Lowest percentage genre: {min_genre} ({min_pct:.2f}%)")

# Plot pie chart
plt.figure(figsize=(10, 8))
plt.pie(
    genre_pct.head(10).values,
    labels=genre_pct.head(10).index,
    autopct="%1.1f%%",
    startangle=90,
    pctdistance=0.6,
)
plt.title("Percentage of Movies by Genre")
plt.axis("equal")
plt.show()

genre_pct.head(10)
%matplotlib inline
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Budget-controlled hypothesis test:
# H0: controlling for budget, franchise status has no relationship with revenue
# H1: franchise films out-earn budget-matched standalone films

sub = df[['budget', 'revenue', 'belongs_to_collection']].copy().dropna()
sub = sub[(sub['budget'] > 0) & (sub['revenue'] > 0)].copy()

# 1 = belongs to a collection; 0 = standalone
sub['franchise'] = sub['belongs_to_collection'].notna().astype(int)
sub['log_budget'] = np.log1p(sub['budget'])
sub['log_revenue'] = np.log1p(sub['revenue'])

X = sm.add_constant(sub[['log_budget', 'franchise']])
y = sub['log_revenue']
model = sm.OLS(y, X).fit()

print('rows_used =', len(sub))
print('budget_coef =', model.params['log_budget'])
print('franchise_coef =', model.params['franchise'])
print('franchise_p_value =', model.pvalues['franchise'])
print('r_squared =', model.rsquared)

# Approximate multiplicative premium from regression coefficient
prem = (np.exp(model.params['franchise']) - 1) * 100
print('approx_revenue_premium_percent =', round(prem, 4))

model.summary()
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Filter to positive, interpretable records
sub = df[['budget', 'revenue']].copy().dropna()
sub = sub[(sub['budget'] > 0) & (sub['revenue'] > 0)]

x = np.log1p(sub['budget'])
y = np.log1p(sub['revenue'])

res = stats.linregress(x, y)
print('rows_used =', len(sub))
print('slope =', res.slope)
print('intercept =', res.intercept)
print('r =', res.rvalue)
print('p_value =', res.pvalue)
print('stderr =', res.stderr)

# Scatter plot with fitted regression line
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.25, s=12)

x_fit = np.linspace(x.min(), x.max(), 200)
y_fit = res.intercept + res.slope * x_fit
plt.plot(x_fit, y_fit, color='red', linewidth=2, label='Regression line')

plt.title('Log-Budget vs Log-Revenue Regression')
plt.xlabel('log(1 + budget)')
plt.ylabel('log(1 + revenue)')
plt.legend()
plt.tight_layout()
plt.show()

res
%matplotlib inline
import numpy as np
import pandas as pd
import statsmodels.api as sm

sub = df[['budget', 'revenue', 'belongs_to_collection']].copy().dropna()
sub = sub[(sub['budget'] > 0) & (sub['revenue'] > 0)].copy()

sub['franchise'] = sub['belongs_to_collection'].notna().astype(int)
sub['log_budget'] = np.log1p(sub['budget'])
sub['log_revenue'] = np.log1p(sub['revenue'])

X = sm.add_constant(sub[['log_budget', 'franchise']])
y = sub['log_revenue']
model = sm.OLS(y, X).fit()

print('rows_used =', len(sub))
print('budget_coef =', model.params['log_budget'])
print('franchise_coef =', model.params['franchise'])
print('franchise_p_value =', model.pvalues['franchise'])
print('r_squared =', model.rsquared)

prem = (np.exp(model.params['franchise']) - 1) * 100
print('approx_revenue_premium_percent =', round(prem, 4))

model.summary()
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Regression test: runtime_minutes -> revenue
sub = df[['runtime_minutes', 'revenue']].copy().dropna()
sub = sub[(sub['runtime_minutes'] > 0) & (sub['revenue'] > 0)]

x = sub['runtime_minutes']
y = np.log1p(sub['revenue'])

res = stats.linregress(x, y)
print('rows_used =', len(sub))
print('slope =', res.slope)
print('intercept =', res.intercept)
print('r =', res.rvalue)
print('p_value =', res.pvalue)
print('stderr =', res.stderr)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.25, s=12)

x_fit = np.linspace(x.min(), x.max(), 200)
y_fit = res.intercept + res.slope * x_fit
plt.plot(x_fit, y_fit, color='red', linewidth=2, label='Regression line')

plt.title('Runtime Minutes vs Log-Revenue Regression')
plt.xlabel('Runtime Minutes')
plt.ylabel('log(1 + revenue)')
plt.legend()
plt.tight_layout()
plt.show()

res
%matplotlib inline
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Parse genres and create one row per genre per movie
rows = []
for _, row in df[['genres', 'revenue']].iterrows():
    genres = [g.strip() for g in str(row['genres']).split(',') if g.strip()]
    for genre in genres:
        rows.append({'genre': genre, 'revenue': row['revenue']})

genre_df = pd.DataFrame(rows)

# Keep only positive revenue values and use log-transform for a more stable ANOVA
genre_df = genre_df[genre_df['revenue'] > 0].copy()
genre_df['log_revenue'] = np.log1p(genre_df['revenue'])

# Limit to genres with enough observations
valid = genre_df.groupby('genre').filter(lambda x: len(x) >= 50)

# Overall ANOVA
samples = [group['log_revenue'].values for _, group in valid.groupby('genre')]
f_stat, p_value = stats.f_oneway(*samples)

print('Overall ANOVA')
print('genres_used =', len(valid['genre'].unique()))
print('f_stat =', f_stat)
print('p_value =', p_value)
print('\nMean revenue by genre:')
print(valid.groupby('genre')['revenue'].mean().sort_values(ascending=False).head(10))

# Post-hoc Tukey HSD test
label = valid['genre']
value = valid['log_revenue']

posthoc = pairwise_tukeyhsd(endog=value, groups=label, alpha=0.05)
print('\nTukey HSD post-hoc results:')
print(posthoc)

%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Hypothesis test: popularity -> vote_average
sub = df[['popularity', 'vote_average']].copy().dropna()
sub = sub[(sub['popularity'] > 0) & (sub['vote_average'] > 0)].copy()

x = np.log1p(sub['popularity'])
y = sub['vote_average']

res = stats.linregress(x, y)
print('rows_used =', len(sub))
print('slope =', res.slope)
print('intercept =', res.intercept)
print('r =', res.rvalue)
print('p_value =', res.pvalue)
print('stderr =', res.stderr)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.2, s=12)

x_fit = np.linspace(x.min(), x.max(), 200)
y_fit = res.intercept + res.slope * x_fit
plt.plot(x_fit, y_fit, color='red', linewidth=2, label='Regression line')

plt.title('Popularity vs Vote Average Regression')
plt.xlabel('log(1 + popularity)')
plt.ylabel('Vote Average')
plt.legend()
plt.tight_layout()
plt.show()

res
import pandas as pd
import glob

# Find the Excel file automatically
excel_file = glob.glob("tmdb_movies_2024_*.xlsx")[0]
print(f"Loading: {excel_file}")

df = pd.read_excel(excel_file)

# Convert budget/revenue to numeric and remove rows with 0 or missing values
df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")

df = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()

print(f"Rows after filtering: {df.shape[0]:,}")

# Save a cleaned copy
output_file = excel_file.replace(".xlsx", "_filtered.xlsx")
df.to_excel(output_file, index=False)
print(f"Saved cleaned data to {output_file}")
