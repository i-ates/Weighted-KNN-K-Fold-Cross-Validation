import euclideDistance as ed
import numpy as np
def getNeighboursWeightedKNN(trainData, targetRow, k):
    distanceList = []
    dataList = []

    # for each element in trainData
    for i in trainData:
        distance = ed.euclideDistance(targetRow, i)  # distance between target sample and the sample in the trainData
        distanceList.append(distance)
        dataList.append(i)
    distanceList = np.array(distanceList)
    dataList = np.array(dataList)

    index_distance = np.argsort(
        distanceList)  # sort the distanceList and add the sorted elements indexes to index_distance
    dataList = dataList[index_distance]  # sort the dataList according to elements in the index_distance
    neighbours = dataList[:k]  # take the nearest k sample to target sample in to neighbours array

    distancesForNeighbours = []  # distances of the neighbours
    for i in range(k):
        element = distanceList[index_distance[i]]
        distancesForNeighbours.append(element)

    return [neighbours, distancesForNeighbours]


# this function takes the nearest neighbours to target sample and calculates the weights of this samples then
# group them by the their class value then sum their weights and choose the class group which haves the max weight
# then returns the class value.
def weightedKnnClassification(trainData, targetRow, k):
    neighours = getNeighboursWeightedKNN(trainData, targetRow, k)
    weights = {}

    for i in range(k):
        if neighours[0][i][-1] not in weights.keys():
            weights[neighours[0][i][-1]] = 1 / neighours[1][i]
        else:
            weight = weights[neighours[0][i][-1]]
            weight = weight + 1 / neighours[1][i]
            weights[neighours[0][i][-1]] = weight

    max = 0
    for i in range(k):
        if weights[neighours[0][i][-1]] > max:
            max = weights[neighours[0][i][-1]]

    for i in range(k):
        if weights[neighours[0][i][-1]] == max:
            prediction = neighours[0][i][-1]
    return prediction

#this function takes the nearest neighbours to the target sample and for each sample: it's multiply their weight and
#target feature value and sum this values. Then divide this value to the sum of these sample's weights. Then return this.
def weightedKnnRegression(trainData, targetRow, k):
    neighours = getNeighboursWeightedKNN(trainData, targetRow, k)

    prediction = 0
    totalWeight = 0

    for i in range(k):
        prediction = prediction + (neighours[0][i][-1] * (1 / neighours[1][i]))
        totalWeight = totalWeight + (1 / neighours[1][i])

    return prediction / totalWeight