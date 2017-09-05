#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 13:22:44 2017

@author: vishnuhari
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


filename ='clusterorder.dat'
clustervariables = [1,2]
seperator='|'
priceenabled= True
pricecolumn_label = "product_id"
quantitycolumn_label="quantity"
valuecolumn_label="value"
customercolvalue = [1]
headerValue=0

'''
filename ='u1.base'
clustervariables = [1,2]
seperator='\t'
priceenabled= False
pricecolumn_label = "price"
quantitycolumn_label="quantity"
valuecolumn_label="value"
customercolvalue=[0]
headerValue=None
'''

#TODO: Here we need to do the plumbing with cassandra to fetch the entire records.
data = pd.read_table(filename,delimiter=seperator, header=headerValue,encoding='latin-1',engine='python')


if priceenabled:
    data[valuecolumn_label] = data[pricecolumn_label] * data[quantitycolumn_label]
    data = data.groupby(['customer_id','product_id'],as_index=False)['quantity'].sum()

influencers = data.iloc[:,clustervariables].values
Y = data.iloc[:,customercolvalue].values

#products = data["product_id"].unique()
#cust = data["customer_id"].unique()

#label the products or SKU
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lbl = LabelEncoder()
influencers[:,0] = lbl.fit_transform(influencers[:,0])
#ohe = OneHotEncoder(categorical_features=[0])
#influencers = ohe.fit_transform(influencers).toarray()

#dummy variable problem
X = np.array(influencers)


#X = influencers[:,:]

#using kmeans lets review how to split clusters.
#we will first use elbow method to reivew how many is the ideal clusters.
from sklearn.cluster import KMeans
wcss =[]
#assume we make 15 clusters.
kmin=1
kmax =16

for index in range(kmin,kmax):
    kmeans = KMeans(n_clusters=index,init='k-means++',random_state=42,max_iter=350)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


#now render the graph
plt.plot(range(kmin,kmax),wcss)
plt.title('Elbow analysis')
plt.xlabel('K count')
plt.ylabel('WCSS value')
plt.show()

index=0
meanlist = wcss -np.mean(wcss)
for item in meanlist:
    if item > 0:
       index +=1;
#which is the position to pick is the question
index = index + 1
    



#based on this, 3-4 seems to be optimal
kmeans = KMeans(n_clusters=index, init='k-means++',random_state=42)
y_kmeans = kmeans.fit_predict(X)

outputcluster = np.array(np.column_stack((Y,y_kmeans)))
dictall = []
dictall = outputcluster.tolist()

centercolor = 'grey'

widthofcluster = np.shape(kmeans.cluster_centers_)[1]-1

#for i in range(0,index):
#    print("i value is {} & width is {}".format(i,widthofcluster))
#    plt.scatter(X[y_kmeans==i,0],X[y_kmeans ==i,1],s=100, c = colors[i] , 
#                  label = "Cluster {}".format(i+1)  )

plt.scatter(X[y_kmeans==0,0],X[y_kmeans ==0,widthofcluster],s=100, c = 'red' ,label = "Cluster {}".format(1)  )
plt.scatter(X[y_kmeans==1,0],X[y_kmeans ==1,widthofcluster],s=100, c = 'blue' ,label = "Cluster {}".format(2)  )
plt.scatter(X[y_kmeans==2,0],X[y_kmeans ==2,widthofcluster],s=100, c = 'green' ,label = "Cluster {}".format(3)  )
plt.scatter(X[y_kmeans==3,0],X[y_kmeans ==3,widthofcluster],s=100, c = 'yellow' ,label = "Cluster {}".format(3)  )
plt.scatter(X[y_kmeans==4,0],X[y_kmeans ==4,widthofcluster],s=100, c = 'orange' ,label = "Cluster {}".format(3)  )

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s=300, c=centercolor,
            label = 'centroids')
plt.xlabel('Product mix')
plt.ylabel('Value purchase')
plt.legend()
plt.show()







