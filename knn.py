import euclideDistance as ed
import numpy as np
from statistics import mode

# this function returns the nearest k neighbours to target sample
def getNeighboursKNN(trainData, targetRow, k):
    distanceList = []
    dataList = []

    # for each element in trainData
    for i in trainData:
        distance = ed.euclideDistance(targetRow,
                                   i)  # distance between target sample and the sample in the trainData
        distanceList.append(distance)
        dataList.append(i)
    distanceList = np.array(distanceList)
    dataList = np.array(dataList)

    index_distance = np.argsort(
        distanceList)  # sort the distanceList and add the sorted elements indexes to index_distance
    dataList = dataList[index_distance]  # sort the dataList according to elements in the index_distance
    neighbours = dataList[:k]  # take the nearest k sample to target sample in to neighbours array

    return neighbours

#this function takes the nearest neighbours to the target sample and return the most repeated class value.
def knnClassification(trainData, targetRow, k):
    neighours = getNeighboursKNN(trainData, targetRow, k)
    classes = []
    for i in neighours:
        classes.append(i[-1])
    prediction = mode(classes)

    return prediction

#this function takes the nearest neighbours to the target sample and sum their target feature's values and divide
#this value to k value.
def knnRegression(trainData, targetRow, k):
    neighours = getNeighboursKNN(trainData, targetRow, k)

    prediction = 0
    for i in range(k):
        prediction = prediction + neighours[i][-1]

    return prediction / k