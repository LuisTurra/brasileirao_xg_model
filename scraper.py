import pandas as pd
from tqdm import tqdm
import time
from playwright.sync_api import sync_playwright

# ================== CONFIG ==================
TOURNAMENT_ID = 325
SEASON_ID = 72034          # ← 2025 


def get_brasileirao_matches():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)  # mude para False na primeira vez pra ver o navegador
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36'
        )
        
        page = context.new_page()
        
        # Página inicial leve 
        print("🌐 Carregando página do Sofascore para pegar cookies...")
        page.goto("https://www.sofascore.com/", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(6000)  
        
        # Endpoint 
        url = f"https://www.sofascore.com/api/v1/unique-tournament/{TOURNAMENT_ID}/season/{SEASON_ID}/events/last/0"
        print(f"📡 Buscando jogos da temporada {SEASON_ID}...")
        response = context.request.get(url)
        
        if response.status != 200:
            print(f"❌ Erro na API: {response.status} - {response.text()}")
            browser.close()
            return pd.DataFrame()
        
        data = response.json()
        
        matches = []
        for event in data.get('events', []):
            if event.get('status', {}).get('type') == 'finished':
                matches.append({
                    'match_id': event['id'],
                    'home_team': event['homeTeam']['name'],
                    'away_team': event['awayTeam']['name'],
                    'date': pd.to_datetime(event['startTimestamp'], unit='s')
                })
        
        browser.close()
        df = pd.DataFrame(matches)
        print(f"✅ {len(df)} jogos finalizados encontrados!")
        return df

def get_shotmap(context, match_id):
    url = f"https://www.sofascore.com/api/v1/event/{match_id}/shotmap"
    response = context.request.get(url)
    if response.status == 200:
        shots = response.json().get('shotmap', [])
        if shots:
            df = pd.DataFrame(shots)
            df['match_id'] = match_id
            return df
    return pd.DataFrame()

# ========================= EXECUÇÃO =========================
if __name__ == "__main__":
    matches = get_brasileirao_matches()
    
    if matches.empty:
        print("❌ Nenhum jogo encontrado. Tente trocar o SEASON_ID")
    else:
        all_shots = []
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36'
            )
            page = context.new_page()
            page.goto("https://www.sofascore.com/", wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(5000)
            
            for mid in tqdm(matches['match_id'], desc="Baixando shotmaps"):
                shots = get_shotmap(context, mid)
                if not shots.empty:
                    all_shots.append(shots)
                time.sleep(0.65)  
            
            browser.close()
        
        if all_shots:
            df_shots = pd.concat(all_shots, ignore_index=True)
            filename = f'shots_brasileirao_{SEASON_ID}.csv'
            df_shots.to_csv(f'data/{filename}', index=False)
            print(f"🎉 PRONTO! {len(df_shots):,} chutes salvos em data/{filename}")
        else:
            print("Nenhum chute encontrado")