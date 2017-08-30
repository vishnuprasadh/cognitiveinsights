#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 09:28:30 2017

@author: vishnuhari
"""

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib as plt



#TODO: Place where we will get from cassandra.
ratings = pd.read_table('u.data',sep="\t", header=0,encoding='latin-1',engine='python',
                        names=['user','movieid','rating','date'])
#introduce a column to multiply value with item
#ratings['linevalue']=customerdata['quantity'] * customerdata['price']

#Get all products
movies = pd.read_table('u.item',sep='|',header=1,encoding='latin-1',engine='python',
                       names=['movieid','name','info'], usecols=[0,1,4])

#Do a customer pivot table with user and product having 
usermoviepivot= ratings.pivot_table(index="user",columns="movieid",values="rating")
#usermoviepivot = usermoviepivot.fillna(0)

#c= pd.DataFrame()


#Normalize the data in pivot.
#usermoviepivot = (usermoviepivot - usermoviepivot.mean())/ (usermoviepivot.max()-usermoviepivot.min())



K = 2 # no of factors. We can increase this or fit this to proper # later.
gamma = 0.001 #this is size of step we take each time we need to reduce error.
lamda = 0.02 #this is the penalization value for regularization.
N = len(usermoviepivot) # No of users
M = len(usermoviepivot.columns) # of products.
steps = 5 #how many times we loop through.
TopN = 10 # Number of recommendations to show


#this is basically user factor matrix of N users, K factors which is also columns.
#index we use is the index value of usermoviepivot. Note that we also have usermoviepivot index is also userIds which
#is equal to number of users. This we call as user factor matrix(refer we used N here) and is randomly selected.
# we will apply SGD i.e. stochiastic gradient descent to find the lowest error or best fit now.
U = pd.DataFrame(np.random.rand(N,K),index=usermoviepivot.index)

#Now we prepare the Product factor matrix.
P= pd.DataFrame(np.random.rand(M,K),index = usermoviepivot.columns)

Utemp = U
Ptemp = P


#Now we need to use SGD to loop through predefined number of times to minimize the SGD error.
#for every user
for step in range(steps):
    for i in usermoviepivot.index:
        for j in usermoviepivot.columns:
            if (usermoviepivot.loc[i,j] >0):
                #print("i {0} and j {1} for step {2}".format(i,j,step))
                #Note that more customers and products would mean so many more records. 
                errorij = usermoviepivot.loc[i,j]-np.dot(U.loc[i],P.loc[j])
                #print("{} pivot has error - {}".format(usermoviepivot.loc[i,j], errorij))
                #now this error has to be minimized or moved down.
                #gamma is the value by which we move the value of U. The rest is the formula used for derivative of slope
                #lambda is the penalization we apply based on factors.
                U.loc[i] = U.loc[i] + gamma * (errorij * P.loc[j] -lamda * U.loc[i])
                P.loc[j]= P.loc[j] + gamma * (errorij * U.loc[i] - lamda * P.loc[j])
            
           
    #Now we did try to pull down SGD through the slope to minimize the errors.Now check if we are good.
    #if not, do more iterations.
    e =0
    for i in usermoviepivot.index:
        for j in usermoviepivot.columns:
            if usermoviepivot.loc[i,j] > 0:
                #Sum of the square of error in the rating.
                e = e + pow(usermoviepivot.loc[i,j] - np.dot(U.loc[i],P.loc[j]),2) + lamda * (pow(np.linalg.norm(U.loc[i]),2) + pow(np.linalg.norm(P.loc[j]),2))
    
    print("{} is the threshold for - {}".format(e,usermoviepivot.loc[i,j]))
    if e <0.01:
        print("e value hit the mark of < 0.01")
        break     
    print("Completed step {}".format(step))
    
print(U,P)

output = np.dot(U,P.T)


activeuser = 1
predict = pd.DataFrame(np.dot(U.loc[activeuser],P.T),index=P.index,columns=["rating"])
toprecommendation = pd.DataFrame.sort_values(predict,['rating'],ascending=[0])[:TopN]    
toprecommendationname = movies[movies["movieid"].isin(toprecommendation.index)]    
print(toprecommendationname["name"].unique())

productforcustomer =ratings[ratings["user"]==activeuser]["movieid"]
proddistinct = productforcustomer.unique()
filter = movies[movies.movieid.isin(proddistinct)]["name"].unique()
print(filter)

                