import streamlit as st
import pandas as pd
import plotly.express as px
from mplsoccer import Pitch
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="xG Brasileirão 2025", layout="wide")
st.title("🔥 Modelo xG - Brasileirão Série A 2025")

# ====================== FUNÇÃO AUXILIAR ======================
def to_dict_safe(val):
    if isinstance(val, dict): return val
    if isinstance(val, str):
        try: return eval(val)
        except: return {}
    return {}

# ====================== CARREGAR DADOS ======================
@st.cache_data
def load_data():
    df = pd.read_csv('data/shots_with_my_xG.csv')
    
    if 'player' in df.columns:
        df['player'] = df['player'].apply(to_dict_safe)
        df['player'] = df['player'].apply(lambda d: d.get('name') if isinstance(d, dict) else str(d))
    
    df['x'] = pd.to_numeric(df.get('x'), errors='coerce')
    df['y'] = pd.to_numeric(df.get('y'), errors='coerce')
    df['my_xG'] = pd.to_numeric(df['my_xG'], errors='coerce')
    df['xg'] = pd.to_numeric(df.get('xg', df.get('xG')), errors='coerce')
    df['is_goal'] = pd.to_numeric(df.get('is_goal', 0), errors='coerce').fillna(0).astype(int)
    
    return df

df = load_data()

# ====================== CARREGA MODELO ======================
@st.cache_resource
def load_model():
    booster = xgb.Booster()
    booster.load_model('xg_model_brasileirao.json')
    return booster

booster = load_model()

# ====================== FEATURES (mesmas do treino) ======================
features = ['distance', 'angle', 'dist_squared', 'dist_angle', 'dist_to_center',
            'dist_to_goal_line', 'is_header', 'is_home', 'minute'] + \
           [c for c in df.columns if c.startswith(('situation_', 'shotType_'))]

# ====================== SIDEBAR ======================
st.sidebar.header("Filtros")
player = st.sidebar.selectbox(
    "Jogador", 
    ["Todos"] + sorted(df['player'].unique())
)

filtered = df.copy()
if player != "Todos":
    filtered = filtered[filtered['player'] == player]

# ====================== TABS ======================
tab1, tab2, tab3 = st.tabs(["📍 Mapa de Chutes", "📊 Performance", "🔬 Modelo vs Sofascore"])

with tab1:
    st.subheader(f"Mapa de Chutes - {player if player != 'Todos' else 'Todos os jogadores'}")
    
    pitch = Pitch(pitch_type='opta', pitch_color='#0e1117', line_color='white', linewidth=2)
    fig, ax = pitch.draw(figsize=(12, 8))
    
    scatter = pitch.scatter(
        filtered['x'], filtered['y'],
        c=filtered['my_xG'],
        s=filtered['my_xG'] * 600 + 30,
        cmap='plasma',
        ax=ax,
        edgecolors='white',
        linewidth=0.8,
        alpha=0.9
    )
    
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Meu xG', rotation=270, labelpad=20)
    
    st.pyplot(fig)

with tab2:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Chutes", len(filtered))
    col2.metric("Gols", int(filtered['is_goal'].sum()))
    col3.metric("xG Sofascore", round(filtered['xg'].sum(), 2))
    col4.metric("**MEU xG**", round(filtered['my_xG'].sum(), 2), 
                delta=round(filtered['my_xG'].sum() - filtered['xg'].sum(), 2))

    st.subheader("Top 5 Jogadores por meu xG")
    top5 = (filtered.groupby('player')['my_xG'].sum()
            .sort_values(ascending=False).head(5).reset_index())
    st.dataframe(top5, use_container_width=True)

with tab3:
    st.subheader("Feature Importance do Modelo")
    
    # Feature importance do Booster 
    imp_dict = booster.get_score(importance_type='gain')
    imp_df = pd.DataFrame({
        'feature': list(imp_dict.keys()),
        'importance': list(imp_dict.values())
    }).sort_values('importance', ascending=False)
    
    fig = px.bar(imp_df.head(12), x='importance', y='feature', orientation='h',
                 title="Top 12 Features Mais Importantes (Gain)")
    st.plotly_chart(fig, use_container_width=True)
    
    st.metric(
        "Correlação meu xG × Sofascore xG",
        round(df['my_xG'].corr(df['xg']), 3)
    )

st.caption("Modelo treinado com 802 chutes da temporada 2025 | Streamlit App")