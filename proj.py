import numpy as np
import pandas as pd

animes = pd.read_csv("cleaned_data.csv")

# Hamming Distance
def hamDist(inputA, inputB):
    pointA = inputA
    pointB = inputB
    total = 0
    if len(pointA.iloc[0]) > len(pointB.iloc[0]):
        if pointB.iloc[0] not in pointA.iloc[0]:
            total += 1
    else:
        if pointA.iloc[0] not in pointB.iloc[0]:
            total += 1
    pointA = pointA.drop(labels='title')
    pointB = pointB.drop(labels='title')
    total += sum(c1 != c2 for c1, c2 in zip(pointA, pointB))
    return total

# Cosine Similarity
def cos_sim(inputA, inputB):
    tempA = inputA
    tempB = inputB
    tempA = tempA.drop(labels=['title'])
    tempB = tempB.drop(labels=['title'])
    pointA = []
    for index, rows in tempA.iteritems():
        if index != 0:
            pointA.append(rows)
    pointB = []
    for index, rows in tempB.iteritems():
        if index != 0:
            pointB.append(rows)
    pointA = np.asarray(pointA, dtype=int)
    pointB = np.asarray(pointB, dtype=int)
    dot_product = np.dot(pointA, pointB)
    norm_a = np.sqrt(sum(i**2 for i in pointA))
    norm_b = np.sqrt(sum(i**2 for i in pointB))
    return 1 - (dot_product / (norm_a * norm_b + eps))

def knnSetup(k):
    lst = []
    for i in range(k):
        lst.append([100000, 0])
    return lst

def sortFirst(val):
    return val[0]

def printHandler(data):
    print("\nRecommendations")
    for point, i in zip(data, range(len(data))):
        print('\t' + str(i+1) + ". " + point[1].iloc[0])

def optimalChoice(lst):
    weightings = lst[0][1]
    weightings = weightings.drop(labels=['title'])
    for i in range(1, len(lst)):
        temp = lst[i][1].drop(labels=['title'])
        weightings = [sum(x) for x in zip(weightings, temp)]
    weightSum = sum(weightings) + len(weightings)
    weightings = [(x / weightSum) for x in weightings]
    return weightings

def convert(lst):
    classification = [(1 if x != 0 else 0) for x in lst]
    return classification

def hamming(weights, p1, p2):
    p2B = p2
    p2B = p2B.drop(labels=['title'])

    total = sum((c0 if (c1 != c2) else 0) for c0, c1, c2 in zip(weights, p1, p2B))
    return total


k = 11
output = []
eps = .000000001

print('Enter an anime title')
title = input()
title.lower()

selection = animes.loc[animes['title'] == title].iloc[0]

optimal = knnSetup(k)
for index, row in animes.iterrows():
    if selection.iloc[0] != row.iloc[0]:
        distance = hamDist(selection, row)
        if (distance < optimal[k-1][0]):
            optimal[k-1] = [distance, row]
            optimal.sort(key = sortFirst)
weights = optimalChoice(optimal)
classification = convert(weights)
optimal = knnSetup(10)
for index, row in animes.iterrows():
    if selection.iloc[0] != row.iloc[0]:
        distance = hamming(weights, classification, row)
        if (distance < optimal[9][0]):
            optimal[9] = [distance, row]
            optimal.sort(key = sortFirst)

printHandler(optimal)
