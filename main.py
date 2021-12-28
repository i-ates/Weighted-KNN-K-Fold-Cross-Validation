import pandas as pd  #just uses for read csv
import numpy as np  #numpy library used throughout the whole project
import dataPreparation as dp
import kFoldCrossValidationClassification as kfc
import kFoldCrossValidationRegression as kfr

#part1
dfGlass = pd.read_csv("glass.csv")
dfGlass = dfGlass.to_numpy()
dfGlass = dp.dropFeatures(dfGlass, [4])
np.random.shuffle(dfGlass)
print("****** Unnormalized data *****")
dfGlass = np.unique(dfGlass, axis=0)
kfc.kFoldCrossValidationClasification(dfGlass, [1, 3, 5, 7, 9])
dfGlass = dp.normalizeData(dfGlass)
dfGlass = np.unique(dfGlass, axis=0)
print("**************** Normalized Data ************")
kfc.kFoldCrossValidationClasification(dfGlass, [1, 3, 5, 7, 9])

#part2
dfConcrete = pd.read_csv("Concrete_Data_Yeh.csv")
dfConcrete = dfConcrete.to_numpy()
np.random.shuffle(dfConcrete)
print("******* Unnormalized Data **********")
dfConcrete = np.unique(dfConcrete, axis=0)
kfr.kFoldCrossValidationRegression(dfConcrete, [1, 3, 5, 7, 9])
dfConcrete = dp.normalizeData(dfConcrete)
dfConcrete = np.unique(dfConcrete, axis=0)
print("*********** Normalized Data ***********")
kfr.kFoldCrossValidationRegression(dfConcrete, [1, 3, 5, 7, 9])