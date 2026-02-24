import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import brier_score_loss, roc_auc_score
import os

csv_path = 'data/shots_brasileirao_72034.csv'
df = pd.read_csv(csv_path)

print(f"✅ Carregado: {df.shape[0]:,} chutes")

def to_dict_safe(val):
    if isinstance(val, dict): return val
    if isinstance(val, str):
        try: return ast.literal_eval(val)
        except: return {}
    return {}

# Coordenadas
df['playerCoordinates'] = df['playerCoordinates'].apply(to_dict_safe)
df['x'] = df['playerCoordinates'].apply(lambda d: d.get('x') if isinstance(d, dict) else np.nan)
df['y'] = df['playerCoordinates'].apply(lambda d: d.get('y') if isinstance(d, dict) else np.nan)
df = df.drop(columns=['playerCoordinates'], errors='ignore')

df['x'] = pd.to_numeric(df['x'], errors='coerce')
df['y'] = pd.to_numeric(df['y'], errors='coerce')
df = df.dropna(subset=['x', 'y']).copy()

# Features espaciais
def get_distance(x, y): return np.sqrt((100 - x)**2 + (y - 50)**2)
def get_angle(x, y):
    goal_w = 7.32
    a = np.sqrt((100 - x)**2 + (y - 50 - goal_w/2)**2)
    b = np.sqrt((100 - x)**2 + (y - 50 + goal_w/2)**2)
    cos_a = np.clip((a**2 + b**2 - goal_w**2) / (2 * a * b), -1.0, 1.0)
    return np.nan_to_num(np.arccos(cos_a) * 180 / np.pi, nan=0.0)

df['distance'] = get_distance(df['x'], df['y'])
df['angle'] = get_angle(df['x'], df['y'])
df['dist_squared'] = df['distance'] ** 2
df['dist_angle'] = df['distance'] * df['angle']
df['dist_to_center'] = np.abs(df['y'] - 50)
df['dist_to_goal_line'] = 100 - df['x']

# Categoricas pré-shot
df['bodyPart_str'] = df['bodyPart'].apply(to_dict_safe).apply(lambda d: d.get('name') if isinstance(d, dict) else str(d))
df['is_header'] = (df['bodyPart_str'].str.lower() == 'head').astype(int)

df['is_home'] = df['isHome'].astype(int)
df['minute'] = pd.to_numeric(df['time'], errors='coerce').fillna(45)

# Target
df['is_goal'] = df['goalType'].notna().astype(int)

# One-hot 
for col in ['situation', 'shotType']:
    if col in df.columns:
        df[col] = df[col].apply(to_dict_safe).apply(lambda d: d.get('name') if isinstance(d, dict) else str(d))
df = pd.get_dummies(df, columns=['situation', 'shotType'], drop_first=True, dummy_na=True)

# Features finais
features = ['distance', 'angle', 'dist_squared', 'dist_angle', 'dist_to_center',
            'dist_to_goal_line', 'is_header', 'is_home', 'minute'] + \
           [c for c in df.columns if c.startswith(('situation_', 'shotType_'))]

print(f"✅ Usando {len(features)} features puramente pré-shot")

X = df[features].fillna(0)
y = df['is_goal']

# Treino
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.02,
    max_depth=5,
    min_child_weight=7,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    reg_alpha=0.8,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)

model.fit(X_train, y_train)

pred_test = model.predict_proba(X_test)[:, 1]
pred_full = model.predict_proba(X)[:, 1]

print("\n" + "="*80)
print(f"Brier Score:  {brier_score_loss(y_test, pred_test):.5f}")
print(f"ROC-AUC:      {roc_auc_score(y_test, pred_test):.4f}")

col_xg = 'xg' if 'xg' in df.columns else 'xG'
if col_xg in df.columns:
    corr = pd.Series(df[col_xg].values).corr(pd.Series(pred_full))
    print(f"Correlação meu xG × Sofascore xG: {corr:.4f}  ← vai subir agora!")

# Feature Importance
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\n🔝 Top 10 features (sem leakage):")
print(importances.head(10))

plt.figure(figsize=(10, 6))
importances.head(15).plot(kind='barh')
plt.title('Feature Importance - xG sem leakage')
plt.tight_layout()
plt.savefig('feature_importance_xg.png')
plt.show()

# Salvar
df['my_xG'] = pred_full
df.to_csv('data/shots_with_my_xG.csv', index=False)
model.save_model('xg_model_brasileirao.json')

print(f"\n🎉 Modelo LIMPO salvo com {len(df):,} chutes!")