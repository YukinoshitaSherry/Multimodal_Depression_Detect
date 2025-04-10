import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# 模拟数据
data = {
    "Fold": [1, 2, 3, 4, 5],
    "F1-Score": [0.74, 0.72, 0.73, 0.71, 0.75],
    "Recall": [0.88, 0.85, 0.86, 0.83, 0.87],
    "Precision": [0.65, 0.63, 0.64, 0.62, 0.66],
    "AUC-ROC": [0.82, 0.80, 0.81, 0.79, 0.83]
}
df = pd.DataFrame(data)

# 1. 各Fold指标趋势（折线图）
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.lineplot(data=df, x="Fold", y="F1-Score", marker="o", label="F1-Score")
sns.lineplot(data=df, x="Fold", y="Recall", marker="s", label="Recall")
sns.lineplot(data=df, x="Fold", y="Precision", marker="D", label="Precision")
plt.title("Classification Metrics Across Folds", fontsize=12)
plt.ylim(0.5, 1.0)
plt.grid(linestyle="--", alpha=0.7)

plt.subplot(1, 2, 2)
sns.lineplot(data=df, x="Fold", y="AUC-ROC", marker="^", color="purple")
plt.title("AUC-ROC Across Folds", fontsize=12)
plt.ylim(0.7, 0.9)
plt.grid(linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# 2. 均值与标准差（柱状图）
plt.figure(figsize=(10, 6))
metrics = ["F1-Score", "Recall", "Precision", "AUC-ROC"]
means = df.mean().values[1:]
stds = df.std().values[1:]

plt.bar(metrics, means, yerr=stds, capsize=10, alpha=0.7,
        color=["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"])
plt.title("Mean Metrics with Standard Deviation", fontsize=14)
plt.ylim(0.5, 0.9)
plt.ylabel("Score")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# 3. ROC曲线与PR曲线（示例：Fold 1）
# 假设已有预测概率和真实标签
y_true = [1, 0, 1, 0, 1, 0, 1, 1, 0, 1]  # 示例数据
y_prob = [0.8, 0.3, 0.7, 0.4, 0.6, 0.2, 0.9, 0.75, 0.25, 0.85]

# ROC曲线
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

# PR曲线
precision, recall, _ = precision_recall_curve(y_true, y_prob)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(recall, precision, label="PR Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.tight_layout()
plt.show()