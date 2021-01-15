import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import pandas as pd
import random as rand
import csv

#Returns the winner of a match based on the scored goals. "H" if home team wins, "A" if away team wins, "D" if it was a draw.
def match_result(home_goals,away_goals):
    goal_diff = int(home_goals) - int(away_goals)
    if(goal_diff > 0):
        return "H"
    elif(goal_diff == 0):
        return "D"
    else:
        return "A"



def observedValue(results,class_i,fold,each_fold,total_matches):
    y = []
    for i in range(0,fold*each_fold):
        if(class_i == results[i][1]):
            y.append(1)
        else:
            y.append(0)
    for i in range((fold+1)*each_fold,total_matches):
        if(class_i == results[i][1]):
            y.append(1)
        else:
            y.append(0)
    return y

def featureMatrix(company):
    X = []
    for i in range(len(company)):   #company[i][0] stores the match_id.
        home_odd = company[i][1]    #company[i][1] stores odds for home win
        draw_odd = company[i][2]    #company[i][2] stores odds for draw
        away_odd = company[i][3]    #company[i][3] stores odds for away win
        X.append([home_odd,draw_odd,away_odd])
    X = np.array(X)
    X=np.insert(X,0,1.0,axis=1)
    return X


def k_fold_cross_validation(company,k_fold):
    matches = len(company)
    each_fold = int(matches/k_fold)
    num_of_matches = each_fold*k_fold
    training_set = []
    testing_set = []
    for fold in range(k_fold):
        start_test = fold*each_fold
        for m in range(0,num_of_matches,each_fold):
            if(m == start_test):
                f = company[m:m+each_fold]
                testing_set.append(company[m:m+each_fold])
            else:
                if(len(training_set) < fold+1):
                    training_set.append(company[m:m+each_fold])
                else:
                    training_set[fold] += company[m:m+each_fold]
    return training_set,testing_set


def score_weights(test_set,w,fold,match_results):
    outcome = ["H","D","A"]
    correct = 0
    wrong = 0
    step = len(test_set)
    start = fold*step
    stop = start+step
    for m in range(start,stop):
        result = []
        i = m-start
        for k in range(len(w)):
            y = w[k][0] + w[k][1]*test_set[i][1] + w[k][2]*test_set[i][2] + w[k][3]*test_set[i][3]
            result.append((1-y)**2)
        best_fit = result.index(min(result))
        if(outcome[best_fit] == match_results[m][1]):
            correct += 1
        else:
            wrong += 1
    return [correct,wrong]
