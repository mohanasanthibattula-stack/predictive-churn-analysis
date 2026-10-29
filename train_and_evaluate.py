"""
Predictive Churn Analysis - Machine Learning Pipeline
Author: Senior ML & MLOps Specialist (Mohanasanthi Battula)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

# 1. Set working directory and paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "customer_churn_model_v2.csv")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)

# 2. Load dataset
df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset successfully. Shape: {df.shape}")

# 3. Separate features and target
# Target: Churn ('Yes' -> 1, 'No' -> 0)
# Drop CustomerID as identifier
df['Churn_Binary'] = df['Churn'].map({'Yes': 1, 'No': 0})
X = df.drop(columns=['CustomerID', 'Churn', 'Churn_Binary'])
y = df['Churn_Binary']

# Define feature types
categorical_cols = ['Region', 'SubscriptionType', 'ContractType', 'PaymentMethod', 'AutoPay']
numerical_cols = ['Age', 'TenureMonths', 'MonthlyChargesINR', 'TotalChargesINR', 'SupportTickets', 'LatePayments']

print(f"Categorical features: {categorical_cols}")
print(f"Numerical features: {numerical_cols}")
print(f"Target distribution:\n{y.value_counts(normalize=True)}")

# 4. ColumnTransformer Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ]
)

# 5. Stratified 80/20 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"Train set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")

# 6. Fit Preprocessor on Training Set
preprocessor.fit(X_train)

# Retrieve feature names after one-hot encoding
cat_encoder = preprocessor.named_transformers_['cat']
cat_encoded_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
all_feature_names = numerical_cols + cat_encoded_feature_names
print(f"Total processed features ({len(all_feature_names)}): {all_feature_names}")

# 7. Model Training & Comparison
# Pipeline 1: Logistic Regression
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
])

# Pipeline 2: Random Forest Classifier
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8, class_weight='balanced'))
])

# Fit models
print("Training Logistic Regression...")
lr_pipeline.fit(X_train, y_train)

print("Training Random Forest Classifier...")
rf_pipeline.fit(X_train, y_train)

# 8. Evaluation on Test Set
models = {
    'Logistic Regression': lr_pipeline,
    'Random Forest': rf_pipeline
}

results = []
preds_dict = {}
probs_dict = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    preds_dict[name] = y_pred
    probs_dict[name] = y_prob
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    })
    
    print(f"\n--- {name} Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['Retained (0)', 'Churned (1)']))
    print(f"ROC-AUC: {roc_auc:.4f}")

results_df = pd.DataFrame(results)
print("\n=== Model Benchmark Comparison ===")
print(results_df.to_string(index=False))

# 9. Visualizations
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# 9.1 ROC-AUC Curves
plt.figure(figsize=(8, 6), dpi=300)
for name in models.keys():
    fpr, tpr, _ = roc_curve(y_test, probs_dict[name])
    auc_val = roc_auc_score(y_test, probs_dict[name])
    plt.plot(fpr, tpr, lw=2.5, label=f"{name} (AUC = {auc_val:.3f})")

plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Random Baseline (AUC = 0.500)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold')
plt.title('Receiver Operating Characteristic (ROC) Curve Comparison', fontsize=14, fontweight='bold', pad=14)
plt.legend(loc='lower right', frameon=True, fontsize=11)
plt.tight_layout()
roc_path = os.path.join(VIZ_DIR, 'roc_auc_curve.png')
plt.savefig(roc_path, dpi=300)
plt.close()
print(f"Saved: {roc_path}")

# 9.2 Confusion Matrices Heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

for idx, (name, ax) in enumerate(zip(models.keys(), axes)):
    cm = confusion_matrix(y_test, preds_dict[name])
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues' if idx == 0 else 'Greens',
        cbar=False, ax=ax, annot_kws={'size': 14, 'weight': 'bold'},
        xticklabels=['Retained (0)', 'Churned (1)'],
        yticklabels=['Retained (0)', 'Churned (1)']
    )
    ax.set_title(f'{name}\nConfusion Matrix', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax.set_ylabel('Actual Label', fontsize=11, fontweight='bold')

plt.suptitle('Confusion Matrix Comparison (Test Set)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
cm_path = os.path.join(VIZ_DIR, 'confusion_matrices.png')
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"Saved: {cm_path}")

# 9.3 Feature Importance / Coefficients
rf_model = rf_pipeline.named_steps['classifier']
rf_importances = rf_model.feature_importances_

lr_model = lr_pipeline.named_steps['classifier']
lr_coefs = lr_model.coef_[0]

importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'RF_Importance': rf_importances,
    'LR_Coefficient': lr_coefs
}).sort_values(by='RF_Importance', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

# Random Forest Top Features
sns.barplot(
    data=importance_df.head(10),
    x='RF_Importance',
    y='Feature',
    palette='viridis',
    ax=axes[0]
)
axes[0].set_title('Top 10 Feature Importances (Random Forest)', fontsize=13, fontweight='bold', pad=10)
axes[0].set_xlabel('Gini Importance (Relative Weight)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Feature Name', fontsize=11, fontweight='bold')

# Logistic Regression Top Coefficients
lr_sorted = importance_df.sort_values(by='LR_Coefficient', ascending=False)
top_lr = pd.concat([lr_sorted.head(5), lr_sorted.tail(5)])
colors = ['#d9534f' if val > 0 else '#5cb85c' for val in top_lr['LR_Coefficient']]
sns.barplot(
    data=top_lr,
    x='LR_Coefficient',
    y='Feature',
    palette=colors,
    ax=axes[1]
)
axes[1].axvline(0, color='black', linestyle='--', lw=1)
axes[1].set_title('Top Logistic Regression Coefficients (Log-Odds Impact)', fontsize=13, fontweight='bold', pad=10)
axes[1].set_xlabel('Coefficient Value (+ increases churn risk)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('')

plt.suptitle('Predictive Churn Risk Drivers & Feature Contribution Analysis', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
fi_path = os.path.join(VIZ_DIR, 'feature_importance.png')
plt.savefig(fi_path, dpi=300)
plt.close()
print(f"Saved: {fi_path}")

# 10. Predict Churn Risk Probabilities Across Full Customer Base
# Best model selection based on ROC-AUC
best_model_name = results_df.sort_values(by='ROC-AUC', ascending=False).iloc[0]['Model']
best_pipeline = models[best_model_name]
print(f"\nSelected Best Performing Model for Scoring: {best_model_name}")

full_probs = best_pipeline.predict_proba(X)[:, 1]

def assign_risk_tier(prob):
    if prob >= 0.70:
        return 'High'
    elif prob >= 0.35:
        return 'Medium'
    else:
        return 'Low'

df_scores = pd.DataFrame({
    'CustomerID': df['CustomerID'],
    'ActualChurn': df['Churn'],
    'PredictedProbability': np.round(full_probs, 4),
    'RiskTier': [assign_risk_tier(p) for p in full_probs]
})

scores_csv_path = os.path.join(BASE_DIR, 'customer_churn_risk_scores.csv')
df_scores.to_csv(scores_csv_path, index=False)
print(f"\nExported Risk Scores to: {scores_csv_path}")
print("Risk Tier Distribution:")
print(df_scores['RiskTier'].value_counts())
print("\nRisk Tier Cross-tab with Actual Churn:")
print(pd.crosstab(df_scores['RiskTier'], df_scores['ActualChurn'], margins=True))

# Save metrics summary to CSV for documentation
metrics_csv_path = os.path.join(BASE_DIR, 'model_evaluation_metrics.csv')
results_df.to_csv(metrics_csv_path, index=False)
print(f"Saved evaluation metrics to: {metrics_csv_path}")
