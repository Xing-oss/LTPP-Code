# -*- coding: utf-8 -*-
"""
Title: A robust Survival Prediction Model for Liver Transplantation Including Intrapatient Variability in Tacrolimus Exposure: A pilot study

Description: This script preprocesses data, trains a Random Forest model, calculates and visualizes feature importance, 
and evaluates model performance using ROC curves for internal and external validation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
from matplotlib.font_manager import FontProperties

# Configure font properties for plots
font = FontProperties(weight='bold')

# Load dataset
features = pd.read_csv('data.csv')

# Normalize selected features
selected_features = features.iloc[:, 2:4]
normalized_features = (selected_features - selected_features.min()) / (selected_features.max() - selected_features.min())
features.iloc[:, 2:4] = normalized_features

# Extract labels and features
labels = np.array(features['state'])
features = features.drop('state', axis=1)
feature_list = list(features.columns)
features = np.array(features)

# Split dataset into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25, random_state=42)

# Train Random Forest model
rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(train_features, train_labels)

# Compute feature importance
importances = list(rf.feature_importances_)
feature_importances = sorted([(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)], key=lambda x: x[1], reverse=True)

# Print feature importance
print("Feature Importance Ranking:")
for feature, importance in feature_importances:
    print(f'Variable: {feature:20} Importance: {importance}')

# Sort features by importance
sorted_indices = np.argsort(importances)
sorted_names = [feature_list[i] for i in sorted_indices]
sorted_importances = [importances[i] for i in sorted_indices]

# Plot feature importance as a gradient bar chart
fig, ax = plt.subplots(figsize=(8, 4))
fig.set_facecolor('#FFFFFF')
ax.set_facecolor('#EFEFEF')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Define color gradient
start_color = "#008000"  # Dark Green
end_color = "#FF8080"  # Light Red
cmap = mcolors.LinearSegmentedColormap.from_list("", [start_color, end_color])
colors = cmap(np.linspace(0, 1, len(feature_list)))

# Plot horizontal bar chart
bars = ax.barh(range(len(feature_list)), sorted_importances, color=colors)
for rect in bars:
    ax.text(rect.get_width(), rect.get_y() + rect.get_height() / 2, f"{rect.get_width():.2f}", ha='left', va='center')

ax.set_yticks(range(len(feature_list)))
ax.set_yticklabels(sorted_names)
ax.set_xlabel('Importance', fontproperties=font)
ax.set_ylabel('Features', fontproperties=font)
ax.grid(color='white', linestyle='-', linewidth=0.5, axis='x')
plt.tight_layout()
plt.savefig('feature_importance_plot.tif', dpi=300)
plt.show()

# Perform ROC Curve Analysis
cv = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
mean_fpr = np.linspace(0, 1, 100)
mean_tpr = 0.0
internal_aucs = []

for train_idx, test_idx in cv.split(train_features, train_labels):
    rf.fit(train_features[train_idx], train_labels[train_idx])
    probs = rf.predict_proba(train_features[test_idx])[:, 1]
    fpr, tpr, _ = roc_curve(train_labels[test_idx], probs)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    internal_aucs.append(auc(fpr, tpr))

mean_tpr /= cv.get_n_splits(train_features, train_labels)
internal_auc = auc(mean_fpr, mean_tpr)
print(f"Internal Validation AUC: {internal_auc:.3f}")

# External validation
external_probs = rf.predict_proba(test_features)[:, 1]
fpr_external, tpr_external, _ = roc_curve(test_labels, external_probs)
external_auc = auc(fpr_external, tpr_external)
print(f"External Validation AUC: {external_auc:.3f}")

# Plot ROC Curves
plt.figure(figsize=(10, 5))
plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='Random Guessing')
plt.plot(mean_fpr, mean_tpr, 'r-', label=f'Internal Validation (AUC = {internal_auc:.3f})', lw=2)
plt.plot(fpr_external, tpr_external, 'b-', label=f'External Validation (AUC = {external_auc:.3f})', lw=2)

plt.xlabel('False Positive Rate', fontproperties=font)
plt.ylabel('True Positive Rate', fontproperties=font)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontproperties=font)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('roc_curve_comparison.tif', dpi=300)
plt.show()
