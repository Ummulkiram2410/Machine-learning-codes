import numpy as np
import matplotlib as plt
import pandas as pd

#Importing data set
dataset=pd.read_csv('Market_Basket_Optimisation.csv',header=None)
transactions=[]
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
    
#Training apriori on datasets
from apyori import apriori
rules=apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)
#minsupport is 3*7/total transactions ie 7501(product purchased atleast 3 times a day)

#Visualising
result=list(rules)
results_list = []
 
for i in range(0, len(result)):
 
 results_list.append('RULE:\t' + str(result[i][0]) + '\nSUPPORT:\t' + str(result[i][1]))
