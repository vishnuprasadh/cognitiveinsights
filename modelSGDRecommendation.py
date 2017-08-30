# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib as plt



#TODO: Place where we will get from cassandra.
customerdata = pd.read_table('clusterorder.dat',sep="|", header=0,encoding='latin-1',engine='python')
#introduce a column to multiply value with item
customerdata['linevalue']=customerdata['quantity'] * customerdata['price']

#Get all products
products = pd.read_table('clusterproduct.dat',sep='|',header=0,encoding='latin-1',engine='python')

#Do a customer pivot table with user and product having 
custorderpivot= customerdata.pivot_table(index="customer_id",columns="product_id",values="quantity", aggfunc=np.sum)
#custorderpivot = custorderpivot.fillna(0)

#c= pd.DataFrame()


#Normalize the data in pivot.
#custorderpivot = (custorderpivot - custorderpivot.mean())/ (custorderpivot.max()-custorderpivot.min())



K = 2 # no of factors. We can increase this or fit this to proper # later.
gamma = 0.1 #this is size of step we take each time we need to reduce error.
lamda = 0.1 #this is the penalization value for regularization.
N = len(custorderpivot) # No of users
M = len(custorderpivot.columns) # of products.
steps = 200 #how many times we loop through.
TopN = 10 # Number of recommendations to show


#this is basically user factor matrix of N users, K factors which is also columns.
#index we use is the index value of custorderpivot. Note that we also have custorderpivot index is also userIds which
#is equal to number of users. This we call as user factor matrix(refer we used N here) and is randomly selected.
# we will apply SGD i.e. stochiastic gradient descent to find the lowest error or best fit now.
U = pd.DataFrame(np.random.rand(N,K),index=custorderpivot.index)

#Now we prepare the Product factor matrix.
P= pd.DataFrame(np.random.rand(M,K),index = custorderpivot.columns)

Utemp = U
Ptemp = P


#Now we need to use SGD to loop through predefined number of times to minimize the SGD error.
#for every user
for step in range(steps):
    for i in custorderpivot.index:
        for j in custorderpivot.columns:
            if (custorderpivot.loc[i,j] >0):
                #print("i {} and j {}".format(i,j))
                #Note that more customers and products would mean so many more records. 
                errorij = custorderpivot.loc[i,j]-np.dot(U.loc[i],P.loc[j])
                #print("{} pivot has error - {}".format(custorderpivot.loc[i,j], errorij))
                #now this error has to be minimized or moved down.
                #gamma is the value by which we move the value of U. The rest is the formula used for derivative of slope
                #lambda is the penalization we apply based on factors.
                U.loc[i] = U.loc[i] + gamma * (errorij * P.loc[j] -lamda * U.loc[i])
                P.loc[j]= P.loc[j] + gamma * (errorij * U.loc[i] - lamda * P.loc[j])
            
           
    #Now we did try to pull down SGD through the slope to minimize the errors.Now check if we are good.
    #if not, do more iterations.
    e =0
    for i in custorderpivot.index:
        for j in custorderpivot.columns:
            if custorderpivot.loc[i,j] > 0:
                #Sum of the square of error in the rating.
                e = e + pow(custorderpivot.loc[i,j] - np.dot(U.loc[i],P.loc[j]),2) + lamda * (pow(np.linalg.norm(U.loc[i]),2) + pow(np.linalg.norm(P.loc[j]),2))
    
    print("{} is the threshold for - {}".format(e,custorderpivot.loc[i,j]))
    
    if e <0.01:
        print("e value hit the mark of < 0.01")
        break;     
print(U,P)

output = np.dot(U,P.T)


activeuser = "000000000116"
predict = pd.DataFrame(np.dot(U.loc[activeuser],P.T),index=P.index,columns=["quantity"])
toprecommendation = pd.DataFrame.sort_values(predict,['quantity'],ascending=[0])[:TopN]    
toprecommendationname = products[products["product_id"].isin(toprecommendation.index)]    
print(toprecommendationname["product_name"].unique())

print(customerdata.shape)    
productforcustomer =customerdata[customerdata["customer_id"]=="000000000116"]["product_id"]
proddistinct = productforcustomer.unique()
filter = products[products.product_id.isin(proddistinct)]["product_name"].unique()
print(filter)

                