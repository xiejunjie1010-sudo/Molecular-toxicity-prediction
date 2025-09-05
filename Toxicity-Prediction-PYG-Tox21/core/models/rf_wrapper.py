# core/models/rf_wrapper.py
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RF:
    """
    简易包装：fit(X, y)      X.shape = (n_samples, n_features)
               predict_proba(X)   返回 List[np.ndarray]，与 sklearn 一致
    """
    def __init__(self, n_est=500):
        self.clf = RandomForestClassifier(
            n_estimators=n_est, n_jobs=-1, class_weight='balanced'
        )
    def fit(self, X: np.ndarray, y: np.ndarray):
        # y 需 shape=(n, 12)；RandomForest 多标签要逐列 fit
        self.models = []
        for i in range(y.shape[1]):
            rf = RandomForestClassifier(
                n_estimators=len(self.clf.estimators_ or [] ) or 500,
                n_jobs=-1, class_weight='balanced'
            )
            rf.fit(X, y[:, i])
            self.models.append(rf)
        return self
    def predict_proba(self, X: np.ndarray):
        # 返回 list(len=12) -> (n,2)，兼容 sklearn 风格
        probs = [m.predict_proba(X) for m in self.models]
        return probs
