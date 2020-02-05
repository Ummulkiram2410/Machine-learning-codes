import pandas as pd
 
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

transactions=[]
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
    
 
from fim import eclat #PyFIM module needs to be installed
rules = eclat(tracts = transactions, supp = 3, zmin=2, report='S') #support in %
rules.sort(key = lambda x: x[1], reverse = True) #sorting in descending order