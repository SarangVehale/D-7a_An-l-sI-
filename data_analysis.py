#!/usr/bin/env python3
"""
advanced_inference.py

Performs comprehensive inferential analyses on `output.db`:
 1. Pairwise chi-square tests (with Cramer’s V) for all categorical pairs
 2. T‑tests or ANOVA for `age` vs. every categorical variable
 3. Logistic regressions for each binary outcome against all other factors
 4. PCA on the full set of dummy‑encoded responses
 5. K‑Means clustering on PCA scores

All results are exported as CSVs in a `results/` folder.
"""

import os
import re
import sqlite3
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Create output directory
os.makedirs('results', exist_ok=True)

# 1. Load data from SQLite
conn = sqlite3.connect('output.db')
df = pd.read_sql_query("SELECT * FROM survery_responses", conn)
conn.close()

# 2. Normalize column names to snake_case
clean_cols = []
for col in df.columns:
    c = re.sub(r'[^0-9a-zA-Z]+', '_', col).strip('_').lower()
    clean_cols.append(c)
df.columns = clean_cols

# 3. Drop metadata & free-text comment columns
#    Assume: first col is timestamp, second is consent, last is comments
meta = [df.columns[0], df.columns[1], df.columns[-1]]
df = df.drop(columns=meta)

# 4. Identify numeric vs categorical
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df.columns if c not in numeric_cols]

# 5. Pairwise chi-square & Cramer’s V for all categorical pairs
def cramers_v(conf_mat):
    chi2, _, _, _ = stats.chi2_contingency(conf_mat)
    n = conf_mat.sum().sum()
    r, k = conf_mat.shape
    return np.sqrt(chi2 / (n * (min(r, k)-1)))

chi_results = []
for a, b in np.combinations(cat_cols, 2):
    table = pd.crosstab(df[a], df[b])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    v = cramers_v(table)
    chi_results.append({
        'var1': a, 'var2': b,
        'chi2': round(chi2, 4), 'p_value': round(p, 4),
        'dof': dof, 'cramers_v': round(v, 4)
    })
pd.DataFrame(chi_results).to_csv('results/chi2_pairwise.csv', index=False)

# 6. T-tests / ANOVA: age vs. each categorical
age = numeric_cols[0] if numeric_cols else None
inf_stats = []
if age:
    for cat in cat_cols:
        groups = [grp[age].dropna() for _, grp in df.groupby(cat)]
        levels = df[cat].dropna().unique()
        if len(levels) == 2:
            t, p = stats.ttest_ind(*groups, equal_var=False)
            inf_stats.append({'variable': cat, 'test': 't-test', 'statistic': round(t,4), 'p_value': round(p,4)})
        elif len(levels) > 2:
            f, p = stats.f_oneway(*groups)
            inf_stats.append({'variable': cat, 'test': 'ANOVA', 'statistic': round(f,4), 'p_value': round(p,4)})
    pd.DataFrame(inf_stats).to_csv('results/age_by_category.csv', index=False)

# 7. Logistic regressions for each binary outcome
#    Identify binary categorical columns
binary_cols = [c for c in cat_cols if df[c].nunique(dropna=True) == 2]
other_cols = [c for c in cat_cols if c not in binary_cols]

for outcome in binary_cols:
    # Prepare y
    levels = list(df[outcome].dropna().unique())
    y = df[outcome].map({levels[0]: 0, levels[1]: 1})

    # Prepare X: one-hot encode other predictors
    X = pd.get_dummies(df[other_cols], drop_first=True).astype(float)
    X = X.loc[y.notna()]  # align
    y = y.loc[X.index]

    X = sm.add_constant(X)
    try:
        model = sm.Logit(y, X).fit(disp=False)
        summary = model.summary2().tables[1]
        summary.to_csv(f'results/logistic_{outcome}.csv')
    except Exception as e:
        print(f"Skipping logistic on {outcome}: {e}")

# 8. PCA on all dummy-encoded responses
X_all = pd.get_dummies(df[cat_cols], drop_first=True).astype(float)
scaler = StandardScaler()
Xs = scaler.fit_transform(X_all)

pca = PCA(n_components=min(10, Xs.shape[1]))
pcs = pca.fit_transform(Xs)

pd.DataFrame(pcs,
             columns=[f'PC{i+1}' for i in range(pcs.shape[1])]
            ).to_csv('results/pca_components.csv', index=False)
pd.DataFrame({'explained_variance_ratio': pca.explained_variance_ratio_}
            ).to_csv('results/pca_variance.csv', index=False)

# 9. K-Means clustering on PCA scores
kmeans = KMeans(n_clusters=3, random_state=0).fit(pcs)
pd.DataFrame({'cluster': kmeans.labels_}).to_csv('results/clusters.csv', index=False)

print("Analysis complete. Results saved under ./results/")

