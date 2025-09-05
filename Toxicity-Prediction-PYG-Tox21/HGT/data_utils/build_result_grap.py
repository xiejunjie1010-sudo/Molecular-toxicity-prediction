import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ======== 路径设置 ========
csv_path = Path("../save/kg_fp5_HGT_pos_weight/summary_per_receptor.csv")        # 读取结果
png_path = csv_path.parent / "rgcn_results_all_metrics.png"   # 保存图片

# ======== 读取并处理数据 ========
df = pd.read_csv(csv_path)

# 如果列名大小写/空格有差异，确保对应
metrics = ["BAC", "F1", "AUC", "Toxic_Acc", "NonToxic_Acc"]
df[metrics] = df[metrics].round(3)      # 保留三位小数

# ======== 绘图参数 ========
plt.figure(figsize=(12, 7), dpi=300)
bar_width = 0.15
x = range(len(df))

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#8c564b", "#9467bd"]
labels = ["BAC", "F1", "AUC", "Toxic Acc", "Non-Toxic Acc"]

# 画 5 组柱状
for idx, metric in enumerate(metrics):
    plt.bar([p + (idx - 2) * bar_width for p in x],
            df[metric],
            width=bar_width,
            label=labels[idx],
            color=colors[idx])

# ======== 美化格式 ========
plt.xticks(x, df["Receptor"], rotation=30, ha="right", fontsize=11, fontweight='bold')
plt.ylabel("Score", fontsize=13, fontweight='bold')
plt.ylim(0.4, 1.05)
plt.title("R-GCN Performance on 12 Receptors (3-decimal precision)", fontsize=15, fontweight='bold', pad=12)
plt.grid(axis='y', linestyle='--', linewidth=0.8, alpha=0.5)
plt.legend(frameon=False, fontsize=11, ncol=3, bbox_to_anchor=(0.5, 1.08), loc="center")
for spine in ["top", "right"]:
    plt.gca().spines[spine].set_visible(False)
plt.tight_layout()

# ======== 保存与展示 ========
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"图片已保存至: {png_path}")
