import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime

team_dict = {'Atlanta Hawks': 'ATL',
 'Boston Celtics': 'BOS',
 'Brooklyn Nets': 'BRK',
 'Charlotte Hornets': 'CHO',
 'Chicago Bulls': 'CHI',
 'Cleveland Cavaliers': 'CLE',
 'Dallas Mavericks': 'DAL',
 'Denver Nuggets': 'DEN',
 'Detroit Pistons': 'DET',
 'Golden State Warriors': 'GSW',
 'Houston Rockets': 'HOU',
 'Indiana Pacers': 'IND',
 'Los Angeles Clippers': 'LAC',
 'Los Angeles Lakers': 'LAL',
 'Memphis Grizzlies': 'MEM',
 'Miami Heat': 'MIA',
 'Milwaukee Bucks': 'MIL',
 'Minnesota Timberwolves':  'MIN',
 'New Orleans Pelicans': 'NOP',
 'New York Knicks': 'NYK',
 'Oklahoma City Thunder': 'OKC',
 'Orlando Magic': 'ORL',
 'Philadelphia 76ers': 'PHI',
 'Phoenix Suns': 'PHO',
 'Portland Trail Blazers': 'POR',
 'Sacramento Kings': 'SAC',
 'San Antonio Spurs': 'SAS',
 'Toronto Raptors': 'TOR',
 'Utah Jazz': 'UTA',
 'Washington Wizards': 'WAS'}

teams = [key for key in team_dict]

# Title of the app
st.title("NBA Matchup Predictor")

st.write("Select two different teams from the menus below.")

# Dropdown menu for the first team
home_key = st.selectbox("Select home team:", 
                        options=[None] + teams, 
                        format_func=lambda x: "Select a team" if x is None else x, 
                        key="home_key")

# Dropdown menu for the second team
# Filter out the team already selected in the first menu
visitor_key = st.selectbox(
    "Select visiting team:",
    options=[None] + [team for team in teams if team != home_key],  # Exclude team1
    format_func=lambda x: "Select a team" if x is None else x,
    key="visitor_key"
)

# Display the selected teams
if home_key and visitor_key:
    st.success(f"You selected: {home_key} and {visitor_key}")
else:
    st.write("Please select two teams")
    
    
# Scrape current player stats

def scrape_basic():
    now = datetime.now()
    if now.month >= 10:
        year = now.year + 1
    else:
        year = now.year
        
    url = f'https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    table = soup.find_all("table")[0]
    
    df = pd.read_html(str(table))[0]
    
    return df

def scrape_adv():
    now = datetime.now()
    if now.month >= 10:
        year = now.year + 1
    else:
        year = now.year
        
    url = f'https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    table = soup.find_all("table")[0]
    
    df = pd.read_html(str(table))[0]
    
    return df
    
    
# Button to trigger the model
if st.button("Run Model"):
    
    st.success("Scraping data...")
    
    home_team = team_dict[home_key]
    visitor_team = team_dict[visitor_key]

    player_reg = scrape_basic()
    player_adv = scrape_adv()
    
    player_reg = player_reg.rename(columns={'Year': 'season'})
    
    player_reg = player_reg.drop(['Rk', 'Age', 'Pos', 'G', 'GS', 'Awards'], axis=1)
    player_adv = player_adv.drop(['MP'], axis=1)
    players = pd.merge(player_reg, player_adv, left_on=['Player', 'Team'], right_on=['Player', 'Team'])
    players = players.drop(['Rk', 'Awards'], axis=1)
    
    players = players.loc[player_reg['Team'] != '2TM']
    players = players.loc[player_reg['Team'] != '3TM']
    players = players.loc[player_reg['Team'] != '4TM']
    
    # SCRAPE 8 PLAYERS FROM EACH TEAM
    url = "https://www.espn.com/nba/injuries"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        print("Page fetched successfully!")
    else:
        print(f"Failed to fetch page. Status code: {response.status_code}")
    
    soup = BeautifulSoup(response.content, "html.parser")
    tables = soup.find_all("table")
    
    dataframes = []
    for index, table in enumerate(tables):
        df = pd.read_html(str(table))[0]
        dataframes.append(df)
    
    df = pd.concat(dataframes, ignore_index=True)
    injuries = df.loc[df['STATUS'] == 'Out', 'NAME']
    
    players = players[~players['Player'].isin(injuries)]
    home_starters = players[players['Team'] == home_team].nlargest(5, 'MP')['Player'].tolist()
    home_reserves = players[players['Team'] == home_team].nlargest(8, 'MP').iloc[5:8]['Player'].tolist()
    visitor_starters = players[players['Team'] == visitor_team].nlargest(5, 'MP')['Player'].tolist()
    visitor_reserves = players[players['Team'] == visitor_team].nlargest(8, 'MP').iloc[5:8]['Player'].tolist()
    
    lineup_data = [[home_starters, home_reserves, visitor_starters, visitor_reserves]]
    cols = ['home_starters', 'home_reserves', 'visitor_starters', 'visitor_reserves']
    
    lineup = pd.DataFrame(lineup_data, columns=cols)
    
    
    
    def merge_starters(lineup, players, stats, side):
        col = lineup[f'{side}_starters'].iloc[0]
        expanded_cols = {}
        for stat in stats:
            expanded_cols[stat] = pd.DataFrame(
                [[players.loc[players['Player'] == player, stat].iloc[0] for player in col]],
                columns=[f'{stat}_player{i+1}_{side}' for i in range(5)]
            )
            
        expanded_df = pd.concat(expanded_cols.values(), axis=1).reset_index()
        return expanded_df
    
    
    def merge_reserves(lineup, players, stats, side):
        col = lineup[f'{side}_reserves'].iloc[0]
        expanded_cols = {}
        for stat in stats:
            expanded_cols[stat] = pd.DataFrame(
                [[players.loc[players['Player'] == player, stat].iloc[0] for player in col]],
                columns=[f'{stat}_player{i+1}_{side}' for i in range(5,8)]
            )
            
        expanded_df = pd.concat(expanded_cols.values(), axis=1).reset_index(drop=True)
        return expanded_df
    
    
    stats = ['WS', 'PER', 'TS%', '3P%', 'BPM', 'PTS']
    
    def merge_all(lineup, players, stats):
        hs = merge_starters(lineup, players, stats, 'home')
        vs = merge_starters(lineup, players, stats, 'visitor')
        hr = merge_reserves(lineup, players, stats, 'home')
        vr = merge_reserves(lineup, players, stats, 'visitor')
        
        x = pd.concat([hs, vs, hr, vr], axis=1).drop(['index'], axis=1)
    
        return x
    
    x = merge_all(lineup, players, stats).fillna(0)
    
    st.success(f"Running the model for: {home_key} vs {visitor_key}")
    
    model = joblib.load('model.pkl')
    
    preds = round(model.predict(x)[0])
    
    if preds > 0:
        model_result = f"The predicted winner is: {home_key} by {preds} points." 
        st.write(model_result)
    else:
        model_result = f"The predicted winner is: {visitor_key} by {-preds} points." 
        st.write(model_result)
    
    
    