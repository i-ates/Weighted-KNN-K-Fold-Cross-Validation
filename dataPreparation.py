import numpy as np

#this function drops an unnecessary feature
def dropFeatures(df, dropList):  # df = dataframe , dropList = list of feature's indexes
    # creating a new numpy array with dataframe shape and an empty list to add necessary features
    new_df = np.zeros(shape=(df.shape[0], df.shape[1] - len(dropList)))
    featureList = []
    # for i in  range dataframe columns count
    for i in range(0, df.shape[1]):
        # if feature is necessary
        if i not in dropList:
            feature = []
            # add a feature in every sample to feature list
            for j in range(0, df.shape[0]):
                feature.append(df[j][i])
            # add feature list to featureList list
            featureList.append(feature)
    # featureList[0].size() = 214 and it's includes first feature values of every sample

    # for x in range necessary feature count
    for x in range(0, df.shape[1] - len(dropList)):
        # for y in range sample's count
        for y in range(0, df.shape[0]):
            new_df[y][x] = featureList[x][y]

    return new_df  # new_df not includes non neccessary features


# In this function, we apply the feature normalization to each value in the dataset.
def normalizeData(df):
    normalizedDf = df
    # for i in range feature's count not includes target feature it's the last element of df
    for i in range(0, df.shape[1] - 1):
        feature = []
        # for j in range sample's count
        for j in range(0, df.shape[0]):
            feature.append(df[j][i])
        # feature[0].size() = 214 and it's includes first feature values of every sample

        # for sample's count it calls normalizeFeature func with calculated minimumum val of a
        # feature and max val a feature
        for x in range(0, df.shape[0]):
            normalizedDf[x][i] = normalizeFeature(df[x][i], min(feature), max(feature))

    return normalizedDf

#In this function, we subtract the minimum value in our feature from our value and divide.
def normalizeFeature(val, min, max):
    return (val - min) / (max - min)  # the new value by the difference between the maximum value and the minimum value.