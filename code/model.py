import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
import ast

# PREPROCESSING

games = pd.read_csv('../data/games_with_players.csv')

# Game Data

games['players'] = games['players'].apply(ast.literal_eval)

games['home_starters'] = games['players'].apply(lambda x: x[1][0])
games['home_reserves'] = games['players'].apply(lambda x: x[1][1])
games['visitor_starters'] = games['players'].apply(lambda x: x[0][0])
games['visitor_reserves'] = games['players'].apply(lambda x: x[0][1])

games = games.drop(['Unnamed: 0', 'Attend.', 'Arena', 'players'], axis=1)
games = games.rename(columns={'Home/Neutral': 'home_team', 'Visitor/Neutral': 'visitor_team'})
games['game_id'] = games.index

team_dict = {'Atlanta Hawks': 'ATL',
 'Boston Celtics': 'BOS',
 'Brooklyn Nets': 'BKN',
 'Charlotte Hornets': 'CHA',
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

games['home_team'] = games['home_team'].apply(lambda x: team_dict[x])
games['visitor_team'] = games['visitor_team'].apply(lambda x: team_dict[x])

# Player Data

player_reg = pd.read_csv('../data/PlayerBasic.csv')
player_adv = pd.read_csv('../data/PlayerAdvanced.csv')

# Delete duplicate columns

player_reg = player_reg.rename(columns={'Year': 'season'})
player_reg = player_reg.drop(['Rk', 'Age', 'Pos', 'G', 'GS', 'Awards', 'Player-additional'], axis=1)
player_adv = player_adv.drop(['MP'], axis=1)

player_reg = player_reg.loc[player_reg['Team'] != '2TM']
player_reg = player_reg.loc[player_reg['Team'] != '3TM']
player_reg = player_reg.loc[player_reg['Team'] != '4TM']

# Merge player stats

players = pd.merge(player_reg, player_adv, left_on=['season', 'Player', 'Team'], right_on=['season', 'Player', 'Team'])
players = players.drop(['Rk', 'Awards', 'Player-additional'], axis=1)
players.head()


# Merge player stats with games

def merge_starters(games, players, stats, side):
    exploded_df = games.explode(f'{side}_starters')

    merged_df = exploded_df.merge(players, left_on=[f'{side}_starters', f'{side}_team', 'season'], right_on=['Player', 'Team', 'season'])

    agg_dict = {stat: lambda x: x.tolist() for stat in stats}
    grouped_df = merged_df.groupby('game_id').agg(agg_dict).reset_index()
    games_with_stats = pd.merge(games, grouped_df, on='game_id')

    expanded_columns = {}
    for stat in stats:
        # Ensure that the list in each row corresponds to the 5 players (1 row per game)
        expanded_columns[stat] = pd.DataFrame(
            games_with_stats[stat].tolist(),
            columns=[f'{stat}_player{i+1}_{side}' for i in range(5)]
        )

    expanded_df = pd.concat(expanded_columns.values(), axis=1)
    final_df = pd.concat([games_with_stats.drop(stats, axis=1), expanded_df], axis=1)

    return final_df

def merge_reserves(games, players, stats, side):
    exploded_df = games.explode(f'{side}_reserves')

    merged_df = exploded_df.merge(players, left_on=[f'{side}_reserves', f'{side}_team', 'season'], right_on=['Player', 'Team', 'season'])

    agg_dict = {stat: lambda x: x.tolist() for stat in stats}
    grouped_df = merged_df.groupby('game_id').agg(agg_dict).reset_index()
    games_with_stats = pd.merge(games, grouped_df, on='game_id')

    expanded_columns = {}
    for stat in stats:
        # Ensure that the list in each row corresponds to the 5 players (1 row per game)
        expanded_columns[stat] = pd.DataFrame(
            games_with_stats[stat].tolist(),
            columns=[f'{stat}_player{i+1}_{side}' for i in range(5,8)]
        )

    expanded_df = pd.concat(expanded_columns.values(), axis=1)
    final_df = pd.concat([games_with_stats.drop(stats, axis=1), expanded_df], axis=1)

    return final_df

players.columns

# Pick Stats to Include

stats = ['WS', 'PER', 'TS%', '3P%', 'BPM', 'PTS']

games = merge_starters(games, players, stats, 'home')
games = merge_starters(games, players, stats, 'visitor')
games = merge_reserves(games, players, stats, 'home')
games = merge_reserves(games, players, stats, 'visitor')

games['outcome'] = games['HomeScore'] - games['VisitorScore']
drop_cols = ['Date', 'Start (ET)', 'visitor_team', 'home_team', 'VisitorScore', 'HomeScore', 'season', 'home_starters', 'home_reserves',
       'visitor_starters', 'visitor_reserves', 'game_id']
games = games.drop(drop_cols, axis=1)
games = games.fillna(0)

x = games.drop(['outcome', 'Unnamed: 0'], axis=1)
y = games['outcome']

cols = x.columns

scaler = StandardScaler()
x = scaler.fit_transform(x)
x = pd.DataFrame(x, columns=cols)

# MODELING

lr = LinearRegression()
xgb = XGBRegressor()
lasso = Lasso(alpha=0.2)
ridge = Ridge(alpha=1.0)
svr = SVR(kernel='linear', C=75, epsilon=0.1)

# Cross Validation

model = lasso

cv = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
neg_mse_scores = cross_val_score(model, x, y, scoring=mse_scorer, cv=cv)

mse_scores = -neg_mse_scores

print("MSE Scores for each fold:", mse_scores)
print("Mean MSE:", np.mean(mse_scores))
print("Standard Deviation of MSE:", np.std(mse_scores))

# Train Test Split

model = lasso
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Drop outliers from training data
df_train = pd.concat([x_train, y_train], axis=1)
df_train = df_train[(df_train['outcome'] < 40) & (df_train['outcome'] > -40)]
y_train = df_train['outcome']
x_train = df_train.drop(['outcome'], axis=1)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

model.fit(x_train, y_train)
y_true = y_test
y_pred = model.predict(x_test)


joblib.dump(model, "model.pkl")
print("Model saved to 'model.pkl'")