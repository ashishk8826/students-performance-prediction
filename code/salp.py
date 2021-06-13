import pandas as pd
import numpy  as np
import math
import random
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC 


def target_function():
    return


def initial_position(swarm_size = 5, min_values = [0,0], max_values = [20,20], target_function = target_function):
    position = np.zeros((swarm_size, len(min_values) + 1))
    for i in range(0, swarm_size):
        for j in range(0, len(min_values)):
             position[i,j] = random.uniform(min_values[j], max_values[j])
        position[i,-1] = target_function(position[i,0:position.shape[1]-1])
    return position


def food_position(dimension = 2, target_function = target_function):
    food = np.zeros((1, dimension+1))
    for j in range(0, dimension):
        food[0,j] = 0.0
    food[0,-1] = target_function(food[0,0:food.shape[1]-1])
    return food


def update_food(position, food):
    for i in range(0, position.shape[0]):
        if (food[0,-1] > position[i,-1]):
            for j in range(0, position.shape[1]):
                food[0,j] = position[i,j]
    return food

def update_position(position, food, c1 = 1, min_values = [0,0], max_values = [20,20], target_function = target_function):
    for i in range(0, position.shape[0]):
        if (i <= position.shape[0]/2):
            for j in range (0, len(min_values)):
                c2 = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
                c3 = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
                if (c3 >= 0.5): #c3 < 0.5
                    position[i,j] = np.clip((food[0,j] + c1*((max_values[j] - min_values[j])*c2 + min_values[j])), min_values[j],max_values[j])
                else:
                    position[i,j] = np.clip((food[0,j] - c1*((max_values[j] - min_values[j])*c2 + min_values[j])), min_values[j],max_values[j])                       
        elif (i > position.shape[0]/2 and i < position.shape[0] + 1):
            for j in range (0, len(min_values)):
                position[i,j] = np.clip(((position[i - 1,j] + position[i,j])/2), min_values[j],max_values[j])             
        position[i,-1] = target_function(position[i,0:position.shape[1]-1])         
    return position


def salp_swarm_algorithm(swarm_size = 5, min_values = [0,0], max_values = [20,20], iterations = 50, target_function = target_function):    
    count    = 0
    position = initial_position(swarm_size = swarm_size, min_values = min_values, max_values = max_values, target_function = target_function)
    food     = food_position(dimension = len(min_values), target_function = target_function)
    while (count <= iterations):     
        print("Iteration = ", count, " G3 marks difference= ", food[0,-1]) 
        c1       = 2*math.exp(-(4*(count/iterations))**2)
        food     = update_food(position, food)        
        position = update_position(position, food, c1 = c1, min_values = min_values, max_values = max_values, target_function = target_function)  
        count    = count + 1 
    print(food)    
    return food




def six_hump_camel_back(variables_values = [0, 0]):
    func_value = 4*variables_values[0]**2 - 2.1*variables_values[0]**4 + (1/3)*variables_values[0]**6 + variables_values[0]*variables_values[1] - 4*variables_values[1]**2 + 4*variables_values[1]**4
    return func_value

ssa = salp_swarm_algorithm(swarm_size = 5, min_values = [0,0], max_values = [20,20], iterations = 50, target_function = six_hump_camel_back)


def rosenbrocks_valley(variables_values = [0,0]):
    func_value = 0
    last_x = variables_values[0]
    for i in range(1, len(variables_values)):
        func_value = func_value + (100 * math.pow((variables_values[i] - math.pow(last_x, 2)), 2)) + math.pow(1 - last_x, 2)
    return func_value

ssa = salp_swarm_algorithm(swarm_size = 15, min_values = [0,0], max_values = [20,20], iterations = 50, target_function = rosenbrocks_valley)



df = pd.read_csv("./data/student-mat.csv", sep=";")




def split_data(X, Y):
    return train_test_split(X, Y, test_size=0.7, random_state=30)


def confuse(y_true, y_pred):
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    fpr(cm)
    ffr(cm)


def fpr(confusion_matrix):
    fp = confusion_matrix[0][1]
    tf = confusion_matrix[0][0]
    rate = float(fp) / (fp + tf)
    print("false Pass Rate: ", rate)


def ffr(confusion_matrix):
    ff = confusion_matrix[1][0]
    tp = confusion_matrix[1][1]
    rate = float(ff) / (ff + tp)
    print("false Fail Rate: ", rate)

    return rate


def train_and_score(X, y):
    X_train, X_test, y_train, y_test = split_data(X, y)

    clf = Pipeline([
        ('reduce_dim', SelectKBest(chi2, k=2)),
        ('train', LinearSVC(C=100))
    ])

    scores = cross_val_score(clf, X_train, y_train, cv=5, n_jobs=2)
    print("Salp Swarm Model Accuracy:", np.array(scores).mean())

    clf.fit(X_train, y_train)

    confuse(y_test, clf.predict(X_test))
    print()


def main():
    print("\nStudent Performance Prediction using Salp Swarm")

    class_le = LabelEncoder()
    for column in df[["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian", "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]].columns:
        df[column] = class_le.fit_transform(df[column].values)

    for i, row in df.iterrows():
        if row["G1"] >= 10:
            df["G1"][i] = 1
        else:
            df["G1"][i] = 0

        if row["G2"] >= 10:
            df["G2"][i] = 1
        else:
            df["G2"][i] = 0

        if row["G3"] >= 10:
            df["G3"][i] = 1
        else:
            df["G3"][i] = 0

    
    y = df.pop("G3")

   
    X = df

    print("\n\nSalp Swarm Accuracy Knowing G1 & G2 Scores")
    print("=====================================")
    train_and_score(X, y)

   
    X.drop(["G2"], axis = 1, inplace=True)
    print("\n\nSalp Swarm Accuracy Knowing Only G1 Score")
    print("=====================================")
    train_and_score(X, y)

   
    X.drop(["G1"], axis=1, inplace=True)
    print("\n\nSalp Swarm Accuracy Without Knowing Scores")
    print("=====================================")
    train_and_score(X, y)



main()
