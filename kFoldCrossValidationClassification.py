import  numpy as np
import datetime  #The library we use to calculate the time difference between algorithms
import knn
import wKNN


# This functios does 5foldcrossvalidation for knn and weighted classification. It takes dataFrame as a numpy array
# and takes K parameters as a list. It split the dataFrame to test Data(%20) and train Data(%20). This function splits
# the dataFrame in to 5 part and takes them respectively each part to test data and take rest of them to train data.
# It calculates the prediction and get accuracy for each part and k Params and and avarage accuracy for knn and weighted knn.
# After each part calculations it prints the results.
def kFoldCrossValidationClasification(df, kParams):
    # splitNum = rounded result of to division of sample count to 5. It is  size of each 5 part
    splitNum = df.shape[0] // 5
    for k in kParams:  # for each K parameters
        avgAccKnn = 0  # avarage result of knn classification
        avgAccWeightedKnn = 0  # avarage result of weighted knn classification
        avgCompileTimeKnn = 0  # avarage compile time for knn classification
        avgCompileTimeWeightedKnn = 0  # avarage compile time for weighted knn classification

        # for each splitted part of dataFrame
        for i in range(0, 5):
            trainData = np.copy(df)  # copy of DataFrame
            testData = np.zeros(shape=(splitNum, df.shape[1]))  # creating test Data array
            # copying samples to test Data and add their indexes to indexList
            for y in range(splitNum * i, splitNum * (i + 1)):
                if i != 0:
                    testData[y - (splitNum * i)] = df[y]
                else:
                    testData[y] = df[y]
            # for  j in range to reversed for loop, delete samples in train Data which are added to test Data
            for j in range(splitNum - 1, -1, -1):
                trainData = np.delete(trainData, (j + (splitNum * i)), 0)

            truePredictCount = 0
            print("-----------" + "Classification" + " k = " + str(k) + "-----------")
            print("***KNN***")
            start_time = datetime.datetime.now()

            # for each sample in test Data
            for z in range(0, testData.shape[0]):
                element = knn.knnClassification(trainData, testData[z], k)  # tahmin sonucu
                # if predicted feature's value is equals to test Data feature's value increase truePredictCount +1
                if element == testData[z][-1]:
                    truePredictCount += 1

            end_time = datetime.datetime.now()
            avgCompileTimeKnn = avgCompileTimeKnn + ((end_time - start_time).total_seconds() * 1000)
            print("Set " + str(i) + " accurancy:" + str(
                (truePredictCount / testData.shape[0]) * 100))
            print((end_time - start_time).total_seconds() * 1000)
            # add accuracy to average accuracy
            avgAccKnn = avgAccKnn + ((truePredictCount / testData.shape[0]) * 100)

            print("***WEIGHTED KNN***")
            truePredictCount = 0
            start_time = datetime.datetime.now()
            # for each sample in test Data
            for j in range(0, testData.shape[0]):
                element = wKNN.weightedKnnClassification(trainData, testData[j], k)
                # if predicted feature's value is equals to test Data feature's value increase truePredictCount +1
                if element == testData[j][-1]:
                    truePredictCount += 1
            end_time = datetime.datetime.now()
            print("Set " + str(i) + " accurancy:" + str((truePredictCount / testData.shape[0]) * 100))
            print((end_time - start_time).total_seconds() * 1000)
            avgCompileTimeWeightedKnn = avgCompileTimeWeightedKnn + ((end_time - start_time).total_seconds() * 1000)
            avgAccWeightedKnn = avgAccWeightedKnn + ((truePredictCount / testData.shape[0]) * 100)
        print("")
        print("")
        print("////////////////////////")
        print("KNN = " + str(avgAccKnn / 5))
        print("Average Run Time = " + str(avgCompileTimeKnn / 5))
        print("Weighted KNN = " + str(avgAccWeightedKnn / 5))
        print("Average Run Time = " + str(avgCompileTimeWeightedKnn / 5))
        print("////////////////////////")
        print("")
        print("")

