# Disk-Failure Prediction
**Why:** classical ML on a DevOps-flavored dataset; teaches the full ML workflow without the complexity of deep learning.

**Data:** Backblaze Drive Stats (public SMART data). Start with a few months of one drive model.

**Stack:** `pandas`, `numpy`, `scikit-learn`, `xgboost` or `lightgbm`, `matplotlib`/`seaborn`, `MLflow`, `FastAPI`, `Docker`, `pytest`.

**Mini-learning tasks (first 1–1.5 days):**

- scikit-learn on the Iris/Titanic datasets: pipelines, train/test split, cross-validation
- Implement k-NN and a decision tree from scratch, then compare with scikit-learn
- Metrics by hand: precision, recall, F1, ROC-AUC, PR curve (why accuracy lies on imbalanced data)
- MLflow: log params/metrics/artifacts for 3 toy runs

**Build steps:**

1. EDA: missing values, class imbalance (failures are rare), which SMART attributes matter.
2. Label engineering: "will this drive fail within N days?" (careful about **data leakage**: split by time, not randomly).
3. Baselines: logistic regression → random forest → gradient boosting. Handle imbalance (class weights, threshold tuning).
4. Track every experiment in MLflow; pick a model by PR-AUC/recall at fixed precision.
5. Explain the model: feature importance, SHAP.
6. Serve it with FastAPI (`/predict`, `/health`, `/metrics` for Prometheus), with input validation (pydantic).
7. Dockerize, deploy to k8s (Deployment, Service, ConfigMap, readiness/liveness probes).
8. Add a Prometheus `ServiceMonitor` and a Grafana panel: request latency, prediction distribution.

**Stretch:** retrain job as a k8s `CronJob`; model loaded from a PVC or object storage.
