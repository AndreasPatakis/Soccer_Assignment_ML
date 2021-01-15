import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import soccer_ml_package.functions as smp

# Main

# SETTING UP DATA <>
Match = []
Team_Attributes = []
Match_Results = []
companies_odds = [[]for x in range(4)]

with open("./kaggle_soccer_csv_matlab/Match.csv","r") as match_csv:
    for line in match_csv:
        Match.append(line[:-1].split(","))

with open("./kaggle_soccer_csv_matlab/TeamAttributes.csv","r") as team_attr_csv:
    for line in team_attr_csv:
        Team_Attributes.append(line[:-1].split(","))

del Match[0]

num_of_matches = len(Match)
num_of_teamAttributes = len(Team_Attributes)
num_of_companies = len(companies_odds)

for m in Match:
    home_goals = m[9]
    away_goals = m[10]
    result = smp.match_result(home_goals,away_goals)
    Match_Results.append([int(m[6]),result])            #Match_Results stores the match result for each match_id. Match_Results[match,result]

index = 11
for company in range(num_of_companies):
    for m in range(num_of_matches):
        companies_odds[company].append([int(Match[m][6]),float(Match[m][index]), float(Match[m][index+1]), float(Match[m][index+2])])
    index += 3

# SETTING UP DATA </>


# CALCULATING WEIGHTS <>
k = 10
company_fold_weight = [[]for x in range(num_of_companies)]
training_set = [[]for x in range(num_of_companies)]
testing_set = [[]for x in range(num_of_companies)]
for company in range(num_of_companies):
    training_set_i,testing_set_i = smp.k_fold_cross_validation(companies_odds[company],k)
    training_set_matches = len(training_set_i[company])
    testing_set_matches = len(testing_set_i[company])
    num_of_matches = training_set_matches + testing_set_matches
    for fold in range(k):
        X = smp.featureMatrix(training_set_i[fold])
        fold_weights = [[]for x in range(3)]
        for i,class_i in enumerate(["H","D","A"]):
            y = smp.observedValue(Match_Results,class_i,fold,testing_set_matches,num_of_matches)
            w_i = np.linalg.inv(X.T @ X) @ (X.T @ y)
            fold_weights[i] = w_i
        company_fold_weight[company].append((fold_weights))
    training_set[company] = (training_set_i)
    testing_set[company] = (testing_set_i)


# CALCULATING WEIGHTS </>


# TESTING SET AND EVALUATING BEST WEIGHTS <>

scores = [[]for x in range(num_of_companies)]
W = []

for company in range(num_of_companies):
    for fold in range(k):
        results = smp.score_weights(testing_set[company][fold],company_fold_weight[company][fold],fold,Match_Results)
        scores[company].append(results)


for company,company_score in enumerate(scores):
    max_score = company_score[0][0]
    best_fold = 0
    for fold in range(1,k):
        if(company_score[fold][0] > max_score):
            max_score = company_score[fold][0]
            best_fold = fold

    best_score = (max_score/testing_set_matches)*100                        #Score of the best fold in % percentage (correct_guess/total_matches)
    W.append(company_fold_weight[company][best_fold])                       #Storing weights of the best fold for each betting company"""





# TESTING SET AND EVALUATING BEST WEIGHTS </>

# FOR PLOTTING PURPOSES <>

xx, yy = np.meshgrid(range(10), range(10))
names = ["HOME","DRAW","AWAY"]
colors = ["lightblue","orange","grey"]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(W)):
    z = (-W[i][1]*xx - W[i][2]*yy - W[i][0])/W[i][3]
    ax.plot_surface(xx, yy, z, color = colors[i],alpha = 0.2)

to_scatter_home_wins = []
to_scatter_draw = []
to_scatter_away_wins = []
for i in range(num_of_matches):
    if(Match_Results[i] == "H"):
        to_scatter_home_wins.append([companies_odds[company][i][0],companies_odds[company][i][1],companies_odds[company][i][2]])
    elif(Match_Results[i] == "D"):
        to_scatter_draw.append([companies_odds[company][i][0],companies_odds[company][i][1],companies_odds[company][i][2]])
    else:
        to_scatter_away_wins.append([companies_odds[company][i][0],companies_odds[company][i][1],companies_odds[company][i][2]])

to_scatter_home_wins = np.array(to_scatter_home_wins)
to_scatter_draw = np.array(to_scatter_draw)
to_scatter_away_wins = np.array(to_scatter_away_wins)


ax.scatter(to_scatter_home_wins[:,0],to_scatter_home_wins[:,1],to_scatter_home_wins[:,2],color = "blue",label = "Home Wins")
ax.scatter(to_scatter_draw[:,0],to_scatter_draw[:,1],to_scatter_draw[:,2],color = "red", label = "Draw")
ax.scatter(to_scatter_away_wins[:,0],to_scatter_away_wins[:,1],to_scatter_away_wins[:,2],color = "black", label = "Away Wins")


plt.title("Odds for Home, Away and Draw")
ax.set_xlabel("HOME WINS")
ax.set_ylabel("DRAW")
ax.set_zlabel("AWAY WINS")
plt.legend()
plt.tight_layout()
plt.show()

# FOR PLOTTING PURPOSES </>
