"""
Proof-of-concept: KANBoostClassifier on the TOM TELECOM churn dataset,
benchmarked against CatBoost's known result (Test AUC = 0.6992, full
100K rows / ~100 columns).
"""
import sys
sys.path.insert(0, "/home/claude/kanboost_project")

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from kanboost import KANBoostClassifier

np.random.seed(42)

df = pd.read_csv("/home/claude/churn_features.csv")

# نفس الـ 10 فيتشرز العددية المستخدمة بالتجارب السابقة، بدون فئوية هالمرة
# (proof-of-concept أول، نكبّر تدريجياً)
feature_cols = ['mou_change_pct', 'months', 'totmrc_Mean', 'rev_change_pct',
                 'tenure_years', 'equipment_age_years', 'eqpdays', 'hnd_price',
                 'avgqty', 'mou_Mean']

SAMPLE_SIZE = 8000  # نفس حجم العيّنة السابقة للمقارنة العادلة
df_sample = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

X = df_sample[feature_cols]
y = df_sample['churn'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train2, X_val, y_train2, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

print("Training KANBoostClassifier...")
t0 = time.time()

model = KANBoostClassifier(
    n_estimators=60,
    learning_rate=0.3,
    kan_hidden=4,
    kan_grid=3,
    kan_steps=25,
    early_stopping_rounds=10,
    random_state=42,
    verbose=True,
)
model.fit(X_train2, y_train2, eval_set=(X_val, y_val))

elapsed = time.time() - t0

test_prob = model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, test_prob)

print("=" * 60)
print(f"KANBoostClassifier  Test AUC : {test_auc:.4f}")
print(f"عدد الـ learners الفعلي (بعد early stopping): {model.best_iteration_}")
print(f"زمن التدريب: {elapsed:.1f} ثانية")
print(f"CatBoost (المشروع الأصلي، 100 عمود / 100K صف): 0.6992")
print("=" * 60)

importances = model.feature_importances()
print("\nأهمية الفيتشرز (تقريبية، من معاملات spline):")
for name, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
    print(f"  {name:25s} {imp:.4f}")
