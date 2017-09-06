# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 13:22:44 2017

@author: vishnuhari
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import json
import os

class CustomerClusters:
    
    '''
    We do initialization of variables only if local file is used. If not, cassandra connection should be used.
    '''
    filename ='data/clusterorder.dat'
    seperator='|'
    
    '''label of the product, quantity and also the price * quantity column name which is dervied'''
    productcolumn_label = "product_id"
    quantitycolumn_label="quantity"
    pricecolumn_label="price"
    valuecolumn_label="value"
    customercolumn_label = "customer_id"
    
    '''We are assuming column 1 will be product and quantity or price'''
    clustervariables = [1,2]
    
    '''final value based on purchase and will be handy for only cluster by purchase value'''
    customercolvalue = [1]
    '''Yes use header'''
    headerValue=0
   
    def learnclassification(self,platform='ajio',cluster=None,byPurchaseValue=False, showelbow=False):
        '''
        Input values would be platform, cluster, and byPurchaseValue.
        
        By default, Purchase value is not used instead the product purchased and the quantity of purchase is used.
        If byPurchaseValue is true, then the model uses the price field, adds the total value of purchase by customer
        and uses this for the model.
        
        If you want to Refresh the model and get the latest cluster of Customers instead of a saved model, you can call this function
        and then the function of getCustomerClusters.
        
        If you have showelbow as True then a binary view of chart is sent for display in browser.
        
        This function finally saves the model in a file which will be loaded whenever the same has to be read for cluster analysis.
        '''
        
        #TODO: Here we need to do the plumbing with cassandra to fetch the entire records.
        data = pd.read_table(self.filename,delimiter=self.seperator, header=self.headerValue,encoding='latin-1',engine='python')
        
        #if by value then multiply and add value, later group it.
        if byPurchaseValue:
            data[self.valuecolumn_label] = data[self.pricecolumn_label] * data[self.quantitycolumn_label]
            data = data.groupby([self.customercolumn_label,self.productcolumn_label],as_index=False)[self.valuecolumn_label].sum()
        else:
            data = data.groupby([self.customercolumn_label,self.productcolumn_label],as_index=False)[self.quantitycolumn_label].sum()
        
        #we know always the value of second and third are customer and product.
        influencers = data.iloc[:,self.clustervariables].values
        #the outcome we expect to map is customer hence , the dependant outcome.
        Y = data.iloc[:,self.customercolvalue].values
        
        
        #label the products or SKU - This is required to avoid any issues later.
        from sklearn.preprocessing import LabelEncoder,OneHotEncoder
        lbl = LabelEncoder()
        influencers[:,0] = lbl.fit_transform(influencers[:,0])
        
        #Get the array as X
        X = np.array(influencers)
        
        #using kmeans lets review how to split clusters.
        #we will first use elbow method to reivew how many is the ideal clusters.
        from sklearn.cluster import KMeans
        wcss =[]
        #assume we make 15 clusters and min of 1.- usually, its not more than 5-10.
        kmin=1
        kmax =16
        
        for index in range(kmin,kmax):
            kmeans = KMeans(n_clusters=index,init='k-means++',random_state=42,max_iter=350)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        
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

        #Save this model.
        output = np.array(np.column_stack((Y,y_kmeans))).tolist()
        self._savetofile(output,platform,cluster,byPurchaseValue)
       
        if showelbow:
            #now render the graph
            buffer = BytesIO()
            plt.title('Elbow analysis')
            plt.xlabel('K count')
            plt.ylabel('WCSS value')
            plt.text("Elbow derived here is {}".format(index))
            plt.plot(range(kmin,kmax),wcss)
            plt.savefig(buffer,format='jpeg')
            sbase64 = base64.b64decode(buffer.getvalue()).decode('utf-8').replace('\n','')
            buffer.close()
            return sbase64
            #plt.show()
        
    
    def _savetofile(self,data,platform,cluster,byPurchaseValue):
        '''
        Saves the model into a file so that the data can be used whenever required.
        '''
        filename = self._returnfile(platform,cluster,byPurchaseValue) 
        with open(file=filename,mode='w',encoding='latin-1') as f:
            json.dump(data,f)
            print("Wrote file successfully")
            f.close()
    
    
    def _returnfile(self,platform,cluster,byPurchaseValue):
        '''
        Always returns a consistent logic in the filename based on input param.
        '''
        if cluster == None:
            cluster = "None"
        filename = "cluser_{}_{}_{}.json".format(platform,cluster,int(byPurchaseValue))
        print("File is {}".format(filename))
        return filename
    
    def _openfiledata(self,platform,cluster,byPurchaseValue):
        filedata = ""
        try:
            filename = self._returnfile(platform,cluster,byPurchaseValue)
            
            with open(file= filename,mode='r',encoding='latin-1') as f:
                print("Reading file")
                filedata = json.loads(f.read())
                f.close()
                print("Read file successfully")
        except FileNotFoundError as FileException:
            filedata = "-1"
            print("File {} not found.".format(filename))
        except Exception as ex:
            print( ex)
            
        return filedata
    
    
    def getAllCustomerClusters(self,platform='ajio',cluster=None,byPurchaseValue=False):
        '''
        For the given platform, cluster and by purchase value or units, the clusters for all customers are given.
        '''
        filedata = self._openfiledata(platform,cluster,byPurchaseValue)
        #If filenotfound
        if filename == "-1":
            self.learnclassification(platform,cluster,byPurchaseValue)
            filedata = self._openfiledata(platform,cluster,byPurchaseValue)
            dictdata = dict(filedata)
            
        return filedata
    
    def _groupbydict(dict):
        return
    
    def getCustomerCluster(self,customerId, platform='ajio',cluster=None,byPurchaseValue=False):
        customers = dict(self.getAllCustomerClusters(platform,cluster,byPurchaseValue))
        clusterId = -1
        try:
            clusterId = customers.get(customerId)
        except Exception as ex:
            return ex
        
        return clusterId
        
        
if __name__ == '__main__':
    cluster = CustomerClusters()
    _platform='ajio'
    _cluster='1'
    _byPurchasevalue=False
    _nthcustomer =12
    output = cluster.learnclassification(platform=_platform,cluster=_cluster,byPurchaseValue=_byPurchasevalue)
    print("Output is {}".format(output))
    customers = cluster.getAllCustomerClusters(_platform,_cluster,_byPurchasevalue)
    firstcustomer = customers[_nthcustomer][0]
    clusterId = cluster.getCustomerCluster(firstcustomer,_platform,_cluster,_byPurchasevalue)
    print("Customer {} is in cluser {}".format(firstcustomer,clusterId))
    
    
    
    
