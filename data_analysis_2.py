#!/usr/bin/env python3
"""
advanced_inference.py

Performs an exhaustive inferential analysis on the survey dataset (`output.db`):
 1. Pairwise χ² tests with Cramér’s V for all categorical pairs (with matrix)
 2. Pearson correlation for numeric pairs
 3. Bootstrap confidence intervals for key mean differences
 4. T‑tests/ANOVA: age vs. each categorical variable
 5. Linear regression (OLS) for age on all predictors
 6. Logistic regressions for each binary outcome (penalized to avoid singularities)
 7. Random Forest models (classifier for binary; regressor for age) with feature importances
 8. PCA on dummy-encoded data (components, variance, loadings)
 9. K‑Means clustering with silhouette scores
10. Hierarchical clustering on PCA scores

Outputs all results into CSVs under `results/`.
"""

import os
import re
import sqlite3
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from itertools import combinations

# Suppress known warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=PerfectSeparationWarning)

# 1. Setup
os.makedirs('results', exist_ok=True)
conn = sqlite3.connect('output.db')
df = pd.read_sql_query("SELECT * FROM survery_responses", conn)
conn.close()

# 2. Clean column names to snake_case
clean_cols = []
for col in df.columns:
    c = re.sub(r'[^0-9a-zA-Z]+', '_', col).strip('_').lower()
    clean_cols.append(c)
df.columns = clean_cols

# 3. Drop metadata/free-text (first 2 and last cols)
df = df.drop(columns=[clean_cols[0], clean_cols[1], clean_cols[-1]])

# 4. Identify numeric vs. categorical
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df.columns if c not in numeric_cols]

# 5. χ² tests & Cramér’s V pairwise
def cramers_v(conf_mat):
    chi2, _, _, _ = stats.chi2_contingency(conf_mat)
    n = conf_mat.values.sum()
    r, k = conf_mat.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

chi_results = []
for a, b in combinations(cat_cols, 2):
    table = pd.crosstab(df[a], df[b])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    v = cramers_v(table)
    chi_results.append({
        'var1': a, 'var2': b,
        'chi2': round(chi2,4), 'p_value': round(p,4),
        'dof': dof, 'cramers_v': round(v,4)
    })
pd.DataFrame(chi_results).to_csv('results/chi2_pairwise.csv', index=False)

# Cramér’s V matrix
v_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)
for a in cat_cols:
    for b in cat_cols:
        v_matrix.loc[a, b] = 1.0 if a == b else cramers_v(pd.crosstab(df[a], df[b]))
v_matrix.to_csv('results/cramers_v_matrix.csv')

# 6. Pearson correlations for numeric pairs
if len(numeric_cols) > 1:
    corr = df[numeric_cols].corr().round(4)
    corr.to_csv('results/numeric_correlation_matrix.csv')

# 7. Bootstrap CI for age difference between first two groups of first categorical
age_col = numeric_cols[0] if numeric_cols else None
if age_col and cat_cols:
    groups = [v.dropna().values for _, v in df.groupby(cat_cols[0])[age_col]]
    if len(groups) >= 2:
        a_vals, b_vals = groups[0], groups[1]
        n_boot = 5000
        diffs = np.array([
            np.mean(np.random.choice(a_vals, len(a_vals), replace=True))
            - np.mean(np.random.choice(b_vals, len(b_vals), replace=True))
            for _ in range(n_boot)
        ])
        low, high = np.percentile(diffs, [2.5, 97.5])
        pd.DataFrame({'ci_lower': [round(low,4)], 'ci_upper': [round(high,4)]})\
          .to_csv('results/bootstrap_age_diff_ci.csv', index=False)

# 8. T-tests/ANOVA: age vs. each categorical
inf_stats = []
if age_col:
    for cat in cat_cols:
        data = [grp[age_col].dropna() for _, grp in df.groupby(cat)]
        levels = df[cat].dropna().unique()
        if len(levels) == 2:
            t, p = stats.ttest_ind(*data, equal_var=False)
            inf_stats.append({'variable': cat, 'test': 't-test',
                              'statistic': round(t,4), 'p_value': round(p,4)})
        elif len(levels) > 2:
            f, p = stats.f_oneway(*data)
            inf_stats.append({'variable': cat, 'test': 'ANOVA',
                              'statistic': round(f,4), 'p_value': round(p,4)})
    pd.DataFrame(inf_stats).to_csv('results/age_by_category.csv', index=False)

# 9. OLS: age ~ all dummies (drop zero-variance predictors)
if age_col:
    X = pd.get_dummies(df[cat_cols], drop_first=True).astype(float)
    y = df[age_col]
    X, y = X.align(y, join='inner', axis=0)
    X = X.loc[:, X.var() > 0]
    X = sm.add_constant(X)
    try:
        model = sm.OLS(y, X).fit()
        model.summary2().tables[1].to_csv('results/ols_age_fullmodel.csv')
    except Exception as e:
        print(f"OLS regression failed: {e}")

# 10. Penalized Logistic regressions for binary outcomes
binary_cols = [c for c in cat_cols if df[c].nunique(dropna=True) == 2]
preds = [c for c in cat_cols if c not in binary_cols]
for outcome in binary_cols:
    y = df[outcome].dropna().map({v: i for i, v in enumerate(df[outcome].dropna().unique())})
    X = pd.get_dummies(df[preds], drop_first=True).astype(float).loc[y.index]
    if X.empty or y.empty:
        continue
    clf = LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000)
    try:
        clf.fit(X, y)
        coefs = pd.Series(clf.coef_[0], index=X.columns)
        coefs.round(4).to_csv(f'results/logistic_coef_{outcome}.csv', header=['coef'])
    except Exception as e:
        print(f"Logistic regression failed for {outcome}: {e}")

# 11. Random Forest: classifier for binary outcomes
for outcome in binary_cols:
    y = df[outcome].dropna()
    X = pd.get_dummies(df[preds], drop_first=True).astype(float).loc[y.index]
    if X.empty: continue
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    imp = pd.Series(clf.feature_importances_, index=X.columns)
    imp.sort_values(ascending=False).round(4).to_csv(
        f'results/rf_classifier_importances_{outcome}.csv', header=['importance'])

# 12. Random Forest Regressor for age
if age_col:
    y = df[age_col].dropna()
    X = pd.get_dummies(df[cat_cols], drop_first=True).astype(float).loc[y.index]
    if not X.empty:
        rfr = RandomForestRegressor(n_estimators=100, random_state=42)
        rfr.fit(X, y)
        imp = pd.Series(rfr.feature_importances_, index=X.columns)
        imp.sort_values(ascending=False).round(4).to_csv(
            'results/rf_regressor_importances_age.csv', header=['importance'])

# 13. PCA on dummy data
X_dum = pd.get_dummies(df[cat_cols], drop_first=True).astype(float)
if not X_dum.empty:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dum)
    pca = PCA(n_components=min(10, X_scaled.shape[1]))
    pcs = pca.fit_transform(X_scaled)
    pd.DataFrame(pcs, columns=[f'PC{i+1}' for i in range(pcs.shape[1])]) \
      .to_csv('results/pca_components.csv', index=False)
    pd.DataFrame({'explained_variance_ratio': pca.explained_variance_ratio_}) \
      .to_csv('results/pca_variance_ratio.csv', index=False)
    loadings = pd.DataFrame(pca.components_.T,
                            index=X_dum.columns,
                            columns=[f'PC{i+1}' for i in range(pcs.shape[1])])
    loadings.round(4).to_csv('results/pca_loadings.csv')

# 14. K-Means clustering & silhouette scores
if 'pcs' in locals() and pcs.size:
    sil = []
    for k in range(2,7):
        km = KMeans(n_clusters=k, random_state=42).fit(pcs)
        sil.append({'k': k, 'silhouette': round(silhouette_score(pcs, km.labels_),4)})
    pd.DataFrame(sil).to_csv('results/kmeans_silhouette.csv', index=False)
    km3 = KMeans(n_clusters=3, random_state=42).fit(pcs)
    pd.DataFrame({'cluster': km3.labels_}).to_csv('results/clusters_kmeans_k3.csv', index=False)

# 15. Hierarchical clustering on PCA scores
if 'pcs' in locals() and pcs.size:
    for k in [2,3,4]:
        agg = AgglomerativeClustering(n_clusters=k).fit(pcs)
        pd.DataFrame({'cluster': agg.labels_})\
          .to_csv(f'results/hierarchical_clusters_k{k}.csv', index=False)

print("Advanced inference complete. See ./results/ for outputs.")

