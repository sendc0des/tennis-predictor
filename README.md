DATA - 
columns - 
['tourney_id', 'tourney_name', 'surface', 'draw_size', 'tourney_level',
 'tourney_date', 'match_num', 'winner_id', 'winner_seed', 'winner_entry',
 'winner_name', 'winner_hand', 'winner_ht', 'winner_ioc', 'winner_age',
 'winner_rank', 'winner_rank_points', 'loser_id', 'loser_seed',
 'loser_entry', 'loser_name', 'loser_hand', 'loser_ht', 'loser_ioc',
 'loser_age', 'loser_rank', 'loser_rank_points', 'score', 'best_of',
 'round', 'minutes', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon',
 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df',
 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved',
 'l_bpFaced']

tourney_id = tournament id based on ATP database

tourney_name = name of the city of the played tournament

surface = hard, clay, grass, carpet

draw_size = 128, 64, 32, 16, 8, 4

tourney_level = G (Grand Slam), 'A' (ATP Tour), 'D' (Davis Cup), 'F' (Masters / ATP Finals)

tourney_date = week of the tournament

match_num =

winner_id = winner id based on ATP players database

winner_seed = 1,2....32 based on draw seed

winner_entry = Q, WC, LL, SE

winner_name = Name + Surname

winner_hand = R (Right), L (Left)

winner_ht = heigth

winner_ioc = nazionality

winner_age = age in yy.yy format

winner_rank = ranking based on ATP rankings database

winner_rank_points = ranking points based on ATP rankings database

loser_id = v. winner_id

loser_seed = v. winner_seed

loser_entry = v. winner_entry

loser_name = v. winner_name

loser_hand = v. winner_hand

loser_ht = v. winner_ht

loser_ioc = v. winner_ioc

loser_age = v. winner_age

loser_rank = v. winner_rank

loser_rank_points = v. winner_rank_points

score = score of the match

best_of = 3 or 5

round = R128, R64, R32, R16, QF, SF, F

minutes = match duration

w_ace = winner aces

w_df = winner double faults

w_svpt = winner service points

w_1stIn = winner 1st serve in

w_1stWon = winner 1st serve won

w_2ndWon = winner 2nd serve won

w_SvGms = winner service games

w_bpSaved = winner saved break points

w_bpFaced = winner faced break points

l_ace = loser aces

l_df = loser double faults

l_svpt = loser service points

l_1stIn = loser 1st serve in

l_1stWon = loser 1st serve won

l_2ndWon = loser 2nd serve won

l_SvGms = loser service games

l_bpSaved = loser saved break points

l_bpFaced = loser faced break points


FEATURES TO USE -
already existing -
surface
tourney_level
tourney_date (if we want to split the dataset according to the year)
winner_id, loser_id (unique IDs, see ATP database)
winner_ht, loser_ht -> ht diff
winner_age, loser_age -> age diff
round
w_ace, l_ace -> ace diff
w_df, l_df -> df diff
winner_rank, loser_rank -> rank diff
minutes = match duration
tourney date -> for recent form diff calculation

feature engineering - 
h2h ratio
recent_form (past n number of matches excluding the one to be predicted) -> recent form diff

DATA TO BE CLEANED -
RangeIndex: 75011 entries, 0 to 75010
Data columns (total 16 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   winner_id      75011 non-null  object 
 1   loser_id       75011 non-null  object 
 2   winner_rank    74392 non-null  float64
 3   loser_rank     73500 non-null  float64
 4   winner_age     74945 non-null  float64
 5   loser_age      74788 non-null  float64
 6   winner_ht      73505 non-null  float64
 7   loser_ht       72043 non-null  float64
 8   surface        74868 non-null  object 
 9   tourney_level  75011 non-null  object 
 10  round          75006 non-null  object 
 11  w_ace          68781 non-null  float64
 12  l_ace          68781 non-null  float64
 13  w_df           68781 non-null  float64
 14  l_df           68781 non-null  float64
 15  minutes        68088 non-null  float64

ISSUES (null values) - 
winner_id           0
loser_id            0
winner_rank       619
loser_rank       1511
winner_age         66
loser_age         223
winner_ht        1506
loser_ht         2968
surface           143
tourney_level       0
round               5
w_ace            6230
l_ace            6230
w_df             6230
l_df             6230
minutes          6923

FIX - drop all rows with null values
Index: 66972 entries, 0 to 75010
Data columns (total 11 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   winner_id      66972 non-null  object 
 1   loser_id       66972 non-null  object 
 2   surface        66972 non-null  object 
 3   tourney_level  66972 non-null  object 
 4   round          66972 non-null  object 
 5   minutes        66972 non-null  float64
 6   rank_diff      66972 non-null  float64
 7   age_diff       66972 non-null  float64
 8   height_diff    66972 non-null  float64
 9   ace_diff       66972 non-null  float64
 10  df_diff        66972 non-null  float64

CREATE THREE COLUMNS PLAYER1 PLAYER2 WINNER AND RANDOMLY DISTRIBUTE THE VALUES OF WINNER_ID AND LOSER_ID AMONG PLAYER1 AND PLAYER2
DEFINE H2H, RECENT FORM, WINNER BINARY FUNCTIONS
ONE HOT ENCODING FOR SURFACE, TOURNEY_LEVEL, ROUND
TRAIN XGBOOST MODEL
