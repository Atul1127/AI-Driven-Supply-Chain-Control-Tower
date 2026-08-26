"""Unsupervised supply-chain disruption detection."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/retail_sales_data.csv")
OUTPUT_PATH = Path("data/disruption_detection.csv")
MODEL_RESULTS_PATH = Path("data/disruption_model_comparison.csv")
PCA_PATH = Path("data/disruption_pca.csv")

FEATURES = [
    "average_lead_time", "lead_time_std", "on_time_percentage",
    "defect_percentage", "fill_percentage", "supplier_delay_rate",
    "demand_mean", "demand_std", "demand_cv", "stockout_rate",
]


def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


def build_supplier_features(df):
    required = {
        "supplier", "lead_time_days", "on_time_rate", "defect_rate",
        "supplier_delay", "ordered_qty", "received_qty", "demand", "stockout",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    x = df.groupby("supplier").agg(
        average_lead_time=("lead_time_days", "mean"),
        lead_time_std=("lead_time_days", "std"),
        on_time_percentage=("on_time_rate", "mean"),
        defect_percentage=("defect_rate", "mean"),
        total_delays=("supplier_delay", "sum"),
        supplier_delay_rate=("supplier_delay", "mean"),
        total_ordered=("ordered_qty", "sum"),
        total_received=("received_qty", "sum"),
        demand_mean=("demand", "mean"),
        demand_std=("demand", "std"),
        stockout_rate=("stockout", "mean"),
        observations=("supplier", "size"),
    ).reset_index()

    x["fill_percentage"] = (
        x["total_received"] / x["total_ordered"].replace(0, np.nan) * 100
    ).fillna(0)
    x["demand_cv"] = (
        x["demand_std"] / x["demand_mean"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    for col in ["on_time_percentage", "defect_percentage", "stockout_rate", "supplier_delay_rate"]:
        x[col] *= 100
    x[["lead_time_std", "demand_std"]] = x[["lead_time_std", "demand_std"]].fillna(0)
    return x


def cluster_metrics(X, labels):
    mask = labels != -1
    unique = np.unique(labels[mask])
    if len(unique) < 2 or mask.sum() <= len(unique):
        return np.nan, np.nan, np.nan
    return (
        silhouette_score(X[mask], labels[mask]),
        davies_bouldin_score(X[mask], labels[mask]),
        calinski_harabasz_score(X[mask], labels[mask]),
    )


def compare_clustering(X):
    rows = []
    for k in range(2, min(8, len(X) - 1) + 1):
        labels = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(X)
        s, db, ch = cluster_metrics(X, labels)
        rows.append({"model": "K-Means", "parameters": f"k={k}", "clusters": k,
                     "noise_pct": 0.0, "silhouette": s, "davies_bouldin": db,
                     "calinski_harabasz": ch})

    k = min(4, len(X) - 1)
    for linkage in ["ward", "complete", "average"]:
        labels = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(X)
        s, db, ch = cluster_metrics(X, labels)
        rows.append({"model": "Hierarchical", "parameters": f"linkage={linkage},k={k}",
                     "clusters": k, "noise_pct": 0.0, "silhouette": s,
                     "davies_bouldin": db, "calinski_harabasz": ch})

    min_samples = max(3, min(5, len(X) // 4))
    for eps in [0.5, 0.75, 1.0, 1.25, 1.5]:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        s, db, ch = cluster_metrics(X, labels)
        clusters = len(set(labels)) - (1 if -1 in labels else 0)
        rows.append({"model": "DBSCAN", "parameters": f"eps={eps}", "clusters": clusters,
                     "noise_pct": float((labels == -1).mean() * 100), "silhouette": s,
                     "davies_bouldin": db, "calinski_harabasz": ch})
    return pd.DataFrame(rows)


def choose_best(results):
    valid = results.dropna(subset=["silhouette", "davies_bouldin", "calinski_harabasz"]).copy()
    if valid.empty:
        return "K-Means", 4
    valid["rank_score"] = (
        valid["silhouette"].rank(ascending=False, pct=True)
        + valid["davies_bouldin"].rank(ascending=True, pct=True)
        + valid["calinski_harabasz"].rank(ascending=False, pct=True)
    )
    best = valid.sort_values("rank_score", ascending=False).iloc[0]
    if best["model"] == "K-Means":
        return "K-Means", int(best["clusters"])
    if best["model"] == "Hierarchical":
        return "Hierarchical", int(best["clusters"])
    return "DBSCAN", float(str(best["parameters"]).split("=")[-1])


def fit_cluster(X, model_name, parameter):
    if model_name == "K-Means":
        return KMeans(n_clusters=int(parameter), n_init=20, random_state=42).fit_predict(X)
    if model_name == "Hierarchical":
        return AgglomerativeClustering(n_clusters=int(parameter), linkage="ward").fit_predict(X)
    return DBSCAN(eps=float(parameter), min_samples=max(3, min(5, len(X) // 4))).fit_predict(X)


def run():
    df = load_data()
    supplier = build_supplier_features(df)
    X = supplier[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    results = compare_clustering(X_scaled)
    model_name, parameter = choose_best(results)
    supplier["cluster"] = fit_cluster(X_scaled, model_name, parameter)

    pca = PCA(n_components=2, random_state=42)
    projected = pca.fit_transform(X_scaled)
    pca_output = supplier[["supplier", "cluster"]].copy()
    pca_output["pc1"] = projected[:, 0]
    pca_output["pc2"] = projected[:, 1]
    pca_output["explained_variance_pc1"] = pca.explained_variance_ratio_[0]
    pca_output["explained_variance_pc2"] = pca.explained_variance_ratio_[1]

    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=42, n_jobs=-1)
    supplier["anomaly_label"] = iso.fit_predict(X_scaled)
    supplier["anomaly_score"] = -iso.score_samples(X_scaled)
    supplier["anomaly_status"] = np.where(supplier["anomaly_label"] == -1, "ANOMALY", "NORMAL")

    # Transparent ranking score for operational alert prioritization.
    supplier["disruption_score"] = (
        0.40 * supplier["anomaly_score"].rank(pct=True) * 100
        + 0.20 * supplier["average_lead_time"].rank(pct=True) * 100
        + 0.15 * (100 - supplier["on_time_percentage"]).rank(pct=True) * 100
        + 0.15 * (100 - supplier["fill_percentage"]).rank(pct=True) * 100
        + 0.10 * supplier["defect_percentage"].rank(pct=True) * 100
    ).round(2)
    supplier["disruption_level"] = pd.cut(
        supplier["disruption_score"],
        bins=[-np.inf, 40, 60, 80, np.inf],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    )
    supplier["cluster_model"] = model_name
    supplier["cluster_parameter"] = parameter
    supplier = supplier.sort_values("disruption_score", ascending=False)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    supplier.to_csv(OUTPUT_PATH, index=False)
    results.to_csv(MODEL_RESULTS_PATH, index=False)
    pca_output.to_csv(PCA_PATH, index=False)

    anomalies = int((supplier["anomaly_status"] == "ANOMALY").sum())
    critical = int((supplier["disruption_level"] == "CRITICAL").sum())
    print(f"Best clustering model: {model_name} ({parameter})")
    print(f"Anomalies detected: {anomalies:,}")
    print(f"Critical disruption candidates: {critical:,}")
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Saved: {MODEL_RESULTS_PATH}")
    print(f"Saved: {PCA_PATH}")


if __name__ == "__main__":
    run()
