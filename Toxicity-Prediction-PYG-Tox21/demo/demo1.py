import numpy as np
filename = ["nr-ahr.smiles", "nr-ar.smiles", "nr-ar-lbd.smiles", "nr-aromatase.smiles", "nr-er.smiles", "nr-er-lbd.smiles", "nr-ppar-gamma.smiles",
                           "sr-are.smiles", "sr-atad5.smiles", "sr-hse.smiles", "sr-mmp.smiles", "sr-p53.smiles"]
# filename = ["nr-ahr.smiles", "nr-ar.smiles", "nr-ar-lbd.smiles", "nr-aromatase.smiles"]
path = 'smote_fp_all_XAttr_more_avg_5_gcn_at'
for name in filename:
    name = name.split('.')[0]
    print(f"{name}：auc:{np.mean(np.load(f'../GCN_AT/save/{path}/auc_{name}.npy')):.3f} acc:{np.mean(np.load(f'../GCN_AT/save/{path}/acc_{name}.npy')):.4f}")

    # smote_fp_Morgan_MACCS_5_graphSAGE
    # smote_x_attr_5_graphSAGE
    # smote_x_attr_MaxMin_5_graphSAGE