import os
import pandas as pd
import glob
import numpy as np

# prepare dataframe of matches from the csv files and exclude the ATP database file
all_csv_files = glob.glob("ML/Tennis Predictor/tennis_data/*.csv")
all_csv_files.pop()
matches = pd.concat(pd.read_csv(f) for f in all_csv_files)

# features to include
features_temp = [
    'winner_id', 'loser_id',  
    'winner_rank', 'loser_rank',    
    'winner_age', 'loser_age',    
    'winner_ht', 'loser_ht',
    'winner_hand', 'loser_hand',
    'surface',
    'tourney_level',
    'round',
    'w_ace', 'l_ace',
    'w_df', 'l_df',
    'w_svpt', 'l_svpt',
    'w_1stWon', 'l_1stWon',
    'w_SvGms', 'l_SvGms',
    'minutes', 'tourney_date'
]
matches = matches[features_temp]

# older matches contain NAN values for many features, drop them
matches = matches.dropna()

# rename features
# bring homogeneity in feature names
matches = matches.rename(columns={'w_ace':'winner_ace', 'l_ace':'loser_ace', 'w_df':'winner_df', 'l_df':'loser_df', 
                                  'w_svpt':'winner_svpt', 'l_svpt':'loser_svpt', 'w_1stWon':'winner_1stWon', 'l_1stWon':'loser_1stWon',
                                  'w_SvGms':'winner_SvGms', 'l_SvGms':'loser_SvGms'})

round_mapping = {'R128':'128','R64':'64','R32':'32','R16':'16','QF':'QF','SF':'SF','F':'F'}
matches['round'] = matches['round'].replace(round_mapping)

# for every match, randomly assign player 1 to be the winner or loser and player 2 accordingly
mask = np.random.rand(len(matches)) > 0.5
matches["player1"] = np.where(mask, matches["winner_id"], matches["loser_id"])
matches["player2"] = np.where(mask, matches["loser_id"], matches["winner_id"])

# modify the features according to the new naming convention
stats = ["rank", "age", "ht", "ace", "df", "svpt", "1stWon", "SvGms"]
for stat in stats:
    matches[f"player1_{stat}"] = np.where(mask, matches[f"winner_{stat}"], matches[f"loser_{stat}"])
    matches[f"player2_{stat}"] = np.where(mask, matches[f"loser_{stat}"], matches[f"winner_{stat}"])

# add a new column of winner binary according to whether player 1 won or not
matches["winner_binary"] = np.where(matches["winner_id"] == matches["player1"], 1, 0)

# convert features into difference between player1's stats and player2's stats
matches["rank_diff"] = matches["player1_rank"] - matches["player2_rank"]
matches["age_diff"]  = matches["player1_age"] - matches["player2_age"]
matches["ht_diff"] = matches["player1_ht"] - matches["player2_ht"]
matches["ace_diff"] = matches["player1_ace"] - matches["player2_ace"]
matches["df_diff"] = matches["player1_df"] - matches["player2_df"]
matches["svpt_diff"] = matches["player1_svpt"] - matches["player2_svpt"]
matches["1stWon_diff"] = matches["player1_1stWon"] - matches["player2_1stWon"]
matches["SvGms_diff"] = matches["player1_SvGms"] - matches["player2_SvGms"]

# function to add H2H feature
def insertH2H_winrate_past(matches):
    """
    Adds 'h2h_winrate' = player1's win rate vs player2
    based only on past matches (before the current match).
    """

    # Sort matches by date (very important!)
    matches = matches.sort_values(by='tourney_date').reset_index(drop=True)

    h2h_winrate = []
    history = {}

    for _, row in matches.iterrows():
        p1, p2, winner = row['player1'], row['player2'], row['winner_binary']

        # --- Lookup history ---
        past = history.get((p1, p2), {"wins": 0, "total": 0})
        if past["total"] == 0:
            h2h_winrate.append(0.5)# no history yet
        else:
            h2h_winrate.append(past["wins"] / past["total"])
        # --- Update history AFTER recording winrate ---
        # Case 1: p1 vs p2
        history.setdefault((p1, p2), {"wins": 0, "total": 0})
        history.setdefault((p2, p1), {"wins": 0, "total": 0})

        if winner == 1:  # player1 wins
            history[(p1, p2)]["wins"] += 1
        else:  # player2 wins
            history[(p2, p1)]["wins"] += 1

        history[(p1, p2)]["total"] += 1
        history[(p2, p1)]["total"] += 1

    matches['h2h_winrate'] = h2h_winrate
    return matches

matches = insertH2H_winrate_past(matches)

# function to add recent form diff feature
def add_recent_form_diff(matches, window=5):
    """
    Adds a single feature: recent form difference (player1 - player2).
    Each player's recent form = win ratio of last N matches (excluding current match).
    """

    # Sort chronologically (important!)
    matches = matches.sort_values(by='tourney_date').reset_index(drop=True)

    recent_form_diff = []
    history = {}

    for _, row in matches.iterrows():
        p1, p2, winner = row['player1'], row['player2'], row['winner_binary']

        # --- Player1 ---
        past_p1 = history.get(p1, [])
        if len(past_p1) == 0:
            form_p1 = None
        else:
            last_matches = past_p1[-window:]
            form_p1 = sum(last_matches) / len(last_matches)

        # --- Player2 ---
        past_p2 = history.get(p2, [])
        if len(past_p2) == 0:
            form_p2 = None
        else:
            last_matches = past_p2[-window:]
            form_p2 = sum(last_matches) / len(last_matches)

        # --- Difference ---
        if form_p1 is None or form_p2 is None:
            recent_form_diff.append(None)
        else:
            recent_form_diff.append(form_p1 - form_p2)

        # --- Update history AFTER computing form ---
        history.setdefault(p1, []).append(1 if winner == 1 else 0)
        history.setdefault(p2, []).append(1 if winner == 0 else 0)

    matches['recent_form_diff'] = recent_form_diff

    return matches

matches = add_recent_form_diff(matches)

# divide the dataframe into X and y
X_features = ['surface', 'tourney_level', 'round', 'minutes',
            'rank_diff', 'age_diff',
            'ht_diff', 'ace_diff', 'df_diff', 'svpt_diff', '1stWon_diff', 'SvGms_diff', 'h2h_winrate', 'recent_form_diff']
X = matches[X_features]
y = matches['winner_binary']

# one hot encoding
X = pd.get_dummies(X,
                   columns = ['surface', 'tourney_level', 'round'],
                   drop_first = True
                )

# split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)
X_train_fit, X_eval, y_train_fit, y_eval = train_test_split(X_train, y_train, train_size=0.8, random_state=42, stratify=y_train)

# create model
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators = 500, learning_rate=0.1, verbosity=1, random_state=42, early_stopping_rounds=50)
model.fit(X_train_fit, y_train_fit, eval_set=[(X_eval, y_eval)])

# evaluate on test set
from sklearn.metrics import accuracy_score
print(accuracy_score(model.predict(X_test), y_test))