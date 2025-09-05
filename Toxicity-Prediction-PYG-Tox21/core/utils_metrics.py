"""
core/utils_metrics.py
---------------------
常用评价指标函数

包含：
1) balanced_accuracy     ——  适应类别不平衡，自动跳过单类样本
2) precision_recall_f1   ——  返回 (Precision, Recall, Fβ)
3) toxic_nontoxic_acc    ——  分别给出有毒 / 无毒预测准确率
4) metrics_report        ——  一次性计算六大核心指标（供其他脚本选用）
"""

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, f1_score

# ----------------------------------------------------------------------
# 1. Balanced Accuracy (修复版)
# ----------------------------------------------------------------------
def balanced_accuracy(y_true, y_pred):
    """
    Balanced Accuracy = (Recall + Specificity) / 2

    - 使用 labels=[0,1] 强制 confusion_matrix 输出 2×2，
      即使测试集中只存在单一类别也不会抛异常。
    - 若正类或负类样本数为 0，返回 np.nan，
      上层 DataFrame.mean() 会自动忽略。
    """
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]   # ★ 关键：显式指定标签
    ).ravel()

    # 若分母为 0 → 单一类别 → 返回 NaN
    if (tp + fn) == 0 or (tn + fp) == 0:
        return np.nan

    recall      = tp / (tp + fn)       # TPR
    specificity = tn / (tn + fp)       # TNR
    return (recall + specificity) / 2


# ----------------------------------------------------------------------
# 2. Precision / Recall / Fβ
# ----------------------------------------------------------------------
def precision_recall_f1(y_true, y_pred, *, beta: float = 1.0):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    beta2 = beta * beta
    f_beta = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)
    return precision, recall, f_beta


# ----------------------------------------------------------------------
# 3. 分别统计有毒 / 无毒准确率
# ----------------------------------------------------------------------
def toxic_nontoxic_acc(tp, fp, tn, fn):
    tox_acc    = tp / (tp + fn + 1e-8)
    nontox_acc = tn / (tn + fp + 1e-8)
    return tox_acc, nontox_acc


# ----------------------------------------------------------------------
# 4. 综合报告（可选）
# ----------------------------------------------------------------------
def metrics_report(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    """
    返回 dict: {AUC, PR_AUC, BAC, F1, ToxicAcc, NontoxAcc}
    自动跳过单类受体的 AUC/PR_AUC 计算。
    """
    y_pred = (y_prob >= threshold).astype(int)

    try:
        auc_val  = roc_auc_score(y_true, y_prob)
        pr_auc   = average_precision_score(y_true, y_prob)
    except ValueError:          # 单一类别
        auc_val  = np.nan
        pr_auc   = np.nan

    f1  = f1_score(y_true, y_pred, zero_division=0)
    bac = balanced_accuracy(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tox_acc, nontox_acc = toxic_nontoxic_acc(tp, fp, tn, fn)

    return dict(
        AUC        = auc_val,
        PR_AUC     = pr_auc,
        BAC        = bac,
        F1         = f1,
        ToxicAcc   = tox_acc,
        NontoxAcc  = nontox_acc,
    )
