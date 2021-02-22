from Neural_Networks_Framework import Neural_Network
import numpy as np
import soccer_ml_package.functions as smp
import random

# SETTING UP DATA <>
Matches_From_Csv = []
Matches = []
Match_Results = []
Team_Attributes = []
features = []
companies = ["B365","BW","IW","LB"]

with open("./kaggle_soccer_csv_matlab/Match3.csv","r") as match_csv:
    for line in match_csv:
        Matches_From_Csv.append(line[:-1].split(","))

with open("./kaggle_soccer_csv_matlab/TeamAttributes.csv","r") as match_csv:
    for line in match_csv:
        Team_Attributes.append(line[:-1].split(","))

del Matches_From_Csv[0]
del Team_Attributes[0]

#random.shuffle(Matches_From_Csv)

temp = []
for t in Team_Attributes:
    year = int(t[3].split("-",1)[0])
    next_year = str(year+1)
    season = str(year)+"/"+next_year
    temp.append([season,float(t[2]),float(t[4])/50,float(t[6])/50,float(t[9])/50,float(t[11])/50,float(t[13])/50,float(t[16])/50,float(t[18])/50,float(t[20])/50])

Team_Attributes = temp
num_of_matches = len(Matches_From_Csv)
num_of_companies = len(companies)


for m in Matches_From_Csv:
    Matches.append([m[3],float(m[7]),float(m[8]),m[11],m[12],m[13],m[14],m[15],m[16],m[17],m[18],m[19],m[20],m[21],m[22]])

for i,m in enumerate(Matches):
    index = 0
    season = m[0]
    home_team = m[1]
    away_team = m[2]
    home_attr = 0
    away_attr = 0
    home_goals = Matches_From_Csv[i][9]
    away_goals = Matches_From_Csv[i][10]
    for i,attr in enumerate(Team_Attributes):
        if(attr[1] == home_team):
            if(season == attr[0]):
                home_attr = i
        elif(attr[1] == away_team):
            if(season == attr[0]):
                away_attr = i
    if(home_attr != 0 and away_attr != 0):
        home = Team_Attributes[home_attr]
        away = Team_Attributes[away_attr]
        temp = [home[2],home[3],home[4],home[5],home[6],home[7],
        home[8],home[9],away[2],away[3],away[4],away[5],away[6],away[7],
        away[8],away[9],float(m[3]),float(m[4]),float(m[5]),float(m[6]),
        float(m[7]),float(m[8]),float(m[9]),float(m[10]),float(m[11]),float(m[12]),
        float(m[13]),float(m[14])]
        features.append(temp)
        #Computing the output data
        result = smp.match_result(home_goals,away_goals)
        if(result == "H"):
            output = [1,0,0]
        elif(result == "D"):
            output = [0,1,0]
        else:
            output = [0,0,1]
        Match_Results.append(output)
print("\nFinished setting up data")
# SETTING UP DATA </>
k = 10
training_sets,testing_sets = smp.k_fold_cross_validation(features,k)
training_outputs,testing_outputs = smp.k_fold_cross_validation(Match_Results,k)

inputs = len(features[0]) #28
learning_rate = 0.2
iterations = 8
scores = []

for fold in range(k):
    print("\n\nExamining Fold : ",fold+1,"/",k," for ",iterations," iterations.")
    net = Neural_Network(inputs,[10,3],learning_rate)
    net.train(training_sets[fold],training_outputs[fold],iterations)
    net.test(testing_sets[fold],testing_outputs[fold])
    score = net.get_Eval()
    scores.append(score)

best_fold = scores.index(max(scores)) + 1

print("\nThe most accurate prediction came from fold ",best_fold," with prediction accuracy: ",scores[best_fold],"%")
