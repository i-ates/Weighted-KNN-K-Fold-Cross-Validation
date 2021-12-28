from math import sqrt

#this function calculates the euclide distance between target sample and other row in train set
def euclideDistance(targetRow, otherRow):
    distance = 0
    for i in range(len(targetRow) - 1):
        distance = distance + (targetRow[i] - otherRow[i]) ** 2

    return sqrt(distance)
