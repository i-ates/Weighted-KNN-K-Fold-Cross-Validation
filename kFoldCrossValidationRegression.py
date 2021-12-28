import  numpy as np
import datetime  #The library we use to calculate the time difference between algorithms
import knn
import wKNN


# This functios does 5foldcrossvalidation for knn and weighted regression. It takes dataFrame as a numpy array
# and takes K parameters as a list. It split the dataFrame to test Data(%20) and train Data(%20). This function splits
# the dataFrame in to 5 part and takes them respectively each part to test data and take rest of them to train data.
# It calculates the prediction and get mean absolute error for each part and k Params and and avarage mean absolute error
# for knn and weighted knn. After each part calculations it prints the results.
def kFoldCrossValidationRegression(df, kParams):
    # splitNum = rounded result of to division of sample count to 5. It is  size of each 5 part
    splitNum = df.shape[0] // 5
    for k in kParams:  # for each K parameters
        avgMaeKnn = 0  # avarage mae of knn regression
        avgMaeWeightedKnn = 0  # avarage mae of weighted knn regression
        avgCompileTimeKnn = 0  # avarage compile time for knn regression
        avgCompileTimeWeightedKnn = 0  # avarage compile time for weighted knn regression

        # for each splitted part of dataFrame
        for i in range(0, 5):
            dummyDf = np.copy(df)  # copy of DataFrame
            indexList = []  # creating test Data array
            testData = np.zeros(shape=(splitNum, df.shape[1]))
            trainData = np.zeros(shape=(df.shape[0] - splitNum, df.shape[1]))
            # copying samples to test Data and add their indexes to indexList
            for y in range(splitNum * i, splitNum * (i + 1)):
                if i != 0:
                    testData[y - (splitNum * i)] = df[y]
                else:
                    testData[y] = df[y]
                indexList.append(y)
            # for  j in range to reversed for loop, delete samples in train Data which are added to test Data
            for j in range(len(indexList) - 1, -1, -1):
                dummyDf = np.delete(dummyDf, (j + (splitNum * i)), 0)

            for j in range(0, dummyDf.shape[0]):
                trainData[j] = dummyDf[j]

            absoluteError = 0
            print("-----------" + "Regression" + " k = " + str(k) + "-----------")
            print("***KNN***")
            start_time = datetime.datetime.now()

            # for each sample in test Data
            for z in range(0, testData.shape[0]):
                element = knn.knnRegression(trainData, testData[z], k)
                absoluteError = absoluteError + abs(element - testData[z][-1])

            end_time = datetime.datetime.now()
            print("Set " + str(i) + " MAE " + str(absoluteError / testData.shape[0]))
            avgCompileTimeKnn = avgCompileTimeKnn + ((end_time - start_time).total_seconds() * 1000)
            print((end_time - start_time).total_seconds() * 1000)
            # add mae to averageMae
            avgMaeKnn = avgMaeKnn + (absoluteError / testData.shape[0])

            print("***WEIGHTED KNN***")
            absoluteError = 0
            start_time = datetime.datetime.now()
            # for each sample in test Data
            for j in range(0, testData.shape[0]):
                element = wKNN.weightedKnnRegression(trainData, testData[j], k)
                absoluteError = absoluteError + abs(element - testData[z][-1])

            end_time = datetime.datetime.now()
            print((end_time - start_time).total_seconds() * 1000)
            avgCompileTimeWeightedKnn = avgCompileTimeWeightedKnn + ((end_time - start_time).total_seconds() * 1000)
            print("Set " + str(i) + " MAE " + str(absoluteError / testData.shape[0]))
            # add mae to averageMae
            avgMaeWeightedKnn = avgMaeWeightedKnn + (absoluteError / testData.shape[0])
        print("")
        print("")
        print("////////////////////////")
        print("KNN = " + str(avgMaeKnn / 5))
        print("Run Time = " + str(avgCompileTimeKnn / 5))
        print("Weighted KNN = " + str(avgMaeWeightedKnn / 5))
        print("Run Time = " + str(avgCompileTimeWeightedKnn / 5))
        print("////////////////////////")
        print("")
        print("")