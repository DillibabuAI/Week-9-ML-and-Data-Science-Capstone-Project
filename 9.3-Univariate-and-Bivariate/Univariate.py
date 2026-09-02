import pandas as pd
import numpy as np
class Univariate():
    def quanQual(dataset):
        quan = []
        qual = []
        for columnName in dataset.columns:
            if dataset[columnName].dtype == 'O':
                qual.append(columnName)
            else:
                quan.append(columnName)
        return quan,qual;
    def freqTable(columnName,dataset):
        freqTable = pd.DataFrame(columns = ["Unique_Values","Frequency","Relative_Frequency","Cusum"])
        freqTable["Unique_Values"] = dataset["totalMethods"].value_counts().index
        freqTable["Frequency"] = dataset["totalMethods"].value_counts().values
        freqTable["Relative_Frequency"] = (freqTable["Frequency"]/6052)
        freqTable["Cusum"] = freqTable["Relative_Frequency"].cumsum()
        return freqTable
    def Univariate(dataset,quan):
        descriptive = pd.DataFrame(index=["Mean", "Median", "Mode", "Q1:25%", "Q2:50%", "Q3:75%", "99%", "Q4:100%", "IQR", "1.5rule", "Lesser", "Greater", "Min", "Max", "kurtosis", "skew", "Var", "Std"], columns=quan)
        for columnName in quan:
            descriptive.loc['Mean', columnName] = dataset[columnName].mean()
            descriptive.loc['Median', columnName] = dataset[columnName].median()
            descriptive.loc['Mode', columnName] = dataset[columnName].mode()[0]
            descriptive.loc['Q1:25%', columnName] = dataset[columnName].quantile(0.25)
            descriptive.loc['Q2:50%', columnName] = dataset[columnName].quantile(0.50)
            descriptive.loc['Q3:75%', columnName] = dataset[columnName].quantile(0.75)
            descriptive.loc['99%', columnName] = dataset[columnName].quantile(0.99)
            descriptive.loc['Q4:100%', columnName] = dataset[columnName].max()
            descriptive.loc['IQR', columnName] = descriptive[columnName]['Q3:75%'] - descriptive[columnName]['Q1:25%']
            descriptive.loc['1.5rule', columnName] = 1.5 * descriptive[columnName]['IQR']
            descriptive.loc['Lesser', columnName] = descriptive[columnName]['Q1:25%'] - descriptive[columnName]['1.5rule']
            descriptive.loc['Greater', columnName] = descriptive[columnName]['Q3:75%'] + descriptive[columnName]['1.5rule']
            descriptive.loc['Min', columnName] = dataset[columnName].min()
            descriptive.loc['Max', columnName] = dataset[columnName].max()
            descriptive.loc['kurtosis', columnName] = dataset[columnName].kurtosis()
            descriptive.loc['skew', columnName] = dataset[columnName].skew()
            descriptive.loc['Var', columnName] = dataset[columnName].var()
            descriptive.loc['Std', columnName] = dataset[columnName].std()
        return descriptive
    def replaceOutiler(dataset, quan, descriptive, lesser, greater):
        for columnName in lesser:
            dataset[columnName] = dataset[columnName].astype('float64')
            dataset.loc[dataset[columnName] < descriptive[columnName]["Lesser"], columnName] = descriptive[columnName]["Lesser"]
        for columnName in greater:
            dataset[columnName] = dataset[columnName].astype('float64')
            dataset.loc[dataset[columnName] > descriptive[columnName]["Greater"], columnName] = descriptive[columnName]["Greater"]
    def checkOutlier(dataset, quan, descriptive):
        lesser = []
        greater = []
        for columnName in quan:
            if (descriptive[columnName]["Min"] < descriptive[columnName]["Lesser"]):
                lesser.append(columnName)
            if (descriptive[columnName]["Max"] > descriptive[columnName]["Greater"]):
                greater.append(columnName)
        return lesser, greater