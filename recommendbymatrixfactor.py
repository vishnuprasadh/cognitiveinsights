#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 20:38:56 2017

@author: vishnuhari
"""

from modelConfiguration import ModelConfiguration
import numpy as np
import pandas as pd
class RecommendProductsByMatrixFactor:
    
    U= pd.DataFrame()
    P=pd.DataFrame()
    
    def trainCustomerPurchase(self,appid,clusterid="1",category="All",no_of_factors=2,gamma=0.02,lamda=0.01,steps=100):
        '''
        Uses input params to create a model. Uses Stochiastic Gradient technique to recommend.
        returns: None but saves the model.
        '''
        try:
                
            uniqueidentity = self.__getuniqueid(appid,clusterid,category)
            print(uniqueidentity)
            #TODO: Place where we will get from cassandra.
            orders = pd.read_table('data/clusterorder.dat',sep="|", header=0,encoding='latin-1',engine='python')
            
            if len(orders)>0:
                print("Cluster analysis for {} rows in progress".format(len(orders)))
                custorderpivot = self.__getcustomerorderpivot(orders)
                
                #Get user, product calculated matrix
                self.U,self.P = self.__generateMatrixFactorizationModel(custorderpivot,
                                                              no_of_factors=no_of_factors,
                                                              gamma=gamma,
                                                              lamda = lamda,
                                                              steps=steps)
            else:
                print("No data to do cluster anlayis")
            
        except Exception as ex:
           print(ex)
    

    
    def __getcustomerorderpivot(self,orders):
        
        #introduce a column to multiply value with item
        orders['linevalue']=orders['quantity'] * orders['price']
        
        #Do a customer pivot table with user and product having 
        custorderpivot= orders.pivot_table(index="customer_id",columns="product_id",values="quantity", aggfunc=np.sum)
        
        #Impute & Normalize the data in pivot. -- this is not required for now.
        #custorderpivot = custorderpivot.fillna(0)
        #custorderpivot = (custorderpivot - custorderpivot.mean())/(custorderpivot.max()-custorderpivot.min())
        
        print("Completed Customerorder pivot")
        return custorderpivot

    def __generateMatrixFactorizationModel(self,custorderpivot,no_of_factors=2,gamma=0.02,lamda=0.01,steps=100):
        '''
        Returns a matrix of User and Product based on the no_of_factors i.e. no of latent factors to be considered for Stochiastic Gradient Descent.
        The customerpivot should have the User as Index and Products as columns.
        Assuming no_of_factors being 2 and lets say the users are 10 and products are 20, we would get a U,P matrix which
        would be of shape U = 10 X2 and P = 20 X 2.
        
        The gamma is used as input to adjust the error by the value provided while lamda is used to penalize the model for the
        number of factors we identify.
        '''
        
        #No need to process if pivot is empty
        if len(custorderpivot)<=0: 
            print("No customer order to do any evaluation")
            return 
                
       
        N = len(custorderpivot) # No of users
        M = len(custorderpivot.columns) # of products.
        steps = 50 #how many times we loop through.
        
        #this is basically user factor matrix of N users, K factors which is also columns.
        #index we use is the index value of custorderpivot. Note that we also have custorderpivot index is also userIds which
        #is equal to number of users. This we call as user factor matrix(refer we used N here) and is randomly selected.
        # we will apply SGD i.e. stochiastic gradient descent to find the lowest error or best fit now.
        U = pd.DataFrame(np.random.rand(N,K),index=custorderpivot.index)
        
        #Now we prepare the Product factor matrix.
        P= pd.DataFrame(np.random.rand(M,K),index = custorderpivot.columns)
        
        
        #Now we need to use SGD to loop through predefined number of times to minimize the SGD error.
        #for every user
        for step in range(steps):
            for i in custorderpivot.index:
                for j in custorderpivot.columns:
                    if (custorderpivot.loc[i,j] >0):
                        #print("i {} and j {}".format(i,j))
                        #Note that more customers and products would mean so many more records. 
                        #errorij is error of actual - predicted. Here np.dot is predicted.
                        errorij = custorderpivot.loc[i,j]-np.dot(U.loc[i],P.loc[j])
                        #print("{} pivot has error - {}".format(custorderpivot.loc[i,j], errorij))
                        #now this error has to be minimized or moved down. We do this by moving the U and P down.
                        #gamma is the value by which we move the value of U. The rest is the formula used for derivative of slope
                        #lambda is the penalization we apply based on factors.
                        U.loc[i] = U.loc[i] + gamma * (errorij * P.loc[j] -lamda * U.loc[i])
                        #Note that the value in bracket is basically the partial derivative or the slope
                        P.loc[j]= P.loc[j] + gamma * (errorij * U.loc[i] - lamda * P.loc[j])
                    
                   
            #Now we did try to pull down SGD through the slope to minimize the errors.Now check if we are good.
            #if not, do more iterations. We continue this until we reach the threshold. In this case it is 0.01 which is very low.
            e =0
            for i in custorderpivot.index:
                for j in custorderpivot.columns:
                    if custorderpivot.loc[i,j] > 0:
                        #Sum of the square of error in the total quantity ordered in our case.
                        e = e + pow(custorderpivot.loc[i,j] - np.dot(U.loc[i],P.loc[j]),2) + lamda * (pow(np.linalg.norm(U.loc[i]),2) + pow(np.linalg.norm(P.loc[j]),2))
                        
            if e <0.01:
                print("e value hit the mark of < 0.01")
                break;
        
        return U,P

    
    def __getuniqueid(self,appid,clusterid,category):
        return "{0}-{1}-{2}".format(str(appid),str(clusterid),category)
        

    def getTopRecommendation(self,userId,topN=10):
        '''
        Given a user or customerId, the output provides you the top 10 recommendations.
        THe output would contain SKUID, Name of product, price, image url, url.
        '''
        #TODO: We should load the U and P. For now am loading assuming class has it.
        U = self.U
        P = self.P
        
        predict = pd.DataFrame(np.dot(U.loc[userId],P.T),index=P.index,columns=["quantity"])
        toprecommendation = pd.DataFrame.sort_values(predict,['quantity'],ascending=[0])[:topN]   
        
        #TODO: Load product details from DB.
        products = pd.read_table('data/clusterproduct.dat',sep='|',header=0,encoding='latin-1',engine='python')
        productdetails = products[products["product_id"].isin(toprecommendation.index)]    
        return productdetails
    
    def getTopRecommendationByCategory(self,userId,topN=10):
        #TODO: Need to add the logic of filtering by category on top of products we got.
        return
        

if __name__ == '__main__':
    rec = RecommendProductsByMatrixFactor()
    rec.trainCustomerPurchase("ajio")
    products= rec.getTopRecommendation("000000000106",10)
    print( products["product_name"].unique())
        

