# 🔥 xG Model - Brasileirão Série A 2025

Modelo de Expected Goals (xG) treinado do zero com dados reais do Sofascore (temporada 2025).

### Live: [streamlit]()

## Funcionalidades

- Coleta automática de todos os chutes via Sofascore API (Playwright)
- Modelo XGBoost com features espaciais avançadas (distância, ângulo, interações, minuto do jogo, cabeçada, mando de campo, etc.)
- Dashboard interativo em Streamlit com:
  - Mapa de chutes no campo (tamanho e cor proporcional ao xG)
  - Métricas de performance por jogador/time
  - Feature Importance
  - Comparação com xG oficial do Sofascore

## Resultados do modelo (802 chutes - temporada 2025)

- **Brier Score**: 0.0977
- **ROC-AUC**: 0.7159
- **Correlação com xG Sofascore**: 0.6497

## Estrutura do projeto

brasileirao_xg_model/
├── data/
│   ├── shots_brasileirao_72034.csv          # dados brutos
│   └── shots_with_my_xG.csv                 # dados com my_xG  
├── streamlit_app.py 
├── xg_model_brasileirao.json          
├── requirements.txt
└── README.md

## Como rodar

### 1. Instalar dependências.

    ```bash
    pip install -r requirements.txt
    playwright install chromium


### 2. Rodar o Streamlit

    ```bash
    streamlit run streamlit_app.py

### 3. Baixar dados 

    ```bash
    python scraper.py

