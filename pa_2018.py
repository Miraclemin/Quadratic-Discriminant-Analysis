# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 11:13:23 2018

@author: XieTianwen
"""

import numpy as np
from numpy import linalg

# load vowel data stored in npy
'''
NOTICE:
labels of y are: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
'''
x_test = np.load('x_test.npy')
print('x_test\'s shape: {}'.format(x_test.shape))
y_test = np.load('y_test.npy')
print('y_test\'s shape: {}'.format(y_test.shape))
x_train = np.load('x_train.npy')
print('x_train\'s shape: {}'.format(x_train.shape))
y_train = np.load('y_train.npy')
print('y_train\'s shape: {}'.format(y_train.shape))

pi = 3.1415926  # value of pi

'''
x : m * n matrix
u : 1 * n vector
sigma ： n * n mtrix
result : 1 * m vector
the function accept x,u ,sigma as parameters and return corresponding probability of N(u,sigma)
you can choise use it to claculate probability if you understand what this function is doing 
your choice!
'''
def density(x,u,sigma):
    n = x.shape[1]
    buff = -0.5*((x-u).dot(linalg.inv(sigma)).dot((x-u).transpose()))
    exp = np.exp(buff)
    C = 1 / np.sqrt(np.power(2*pi,n)*linalg.det(sigma))
    result = np.diag(C*exp)
    return result


'''
class GDA
self.X : training data X
self.Y : training label Y
self.is_linear : True for LDA ,False for QDA ,default True
please make youself konw basic konwledge about python class programming

tips : function you may use
np.histogram(bins = self.n_class)
np.reshape()
np.transpose()
np.dot()
np.argmax()

'''
class GDA():
    def __init__(self, X, Y, is_linear = True):
        self.X = X
        self.Y = Y
        self.is_linear =  is_linear
        self.n_class = len(np.unique(y_train)) # number of class , 11 in this problem 
        self.n_feature = self.X.shape[1]       # feature dimention , 10 in this problem
        self.pro = np.zeros(self.n_class)     # variable stores the probability of each class
        self.mean = np.zeros((self.n_class,self.n_feature)) #variable store the mean of each class
        self.sigma = np.zeros((self.n_class,self.n_feature,self.n_feature)) # variable store the covariance of each class
        self.class_num = np.zeros(self.n_class)
    def calculate_pro(self):
        #calculate the probability of each class and store them in  self.pro
        self.pro = np.histogram(self.Y.reshape(1,-1)[0], bins=np.arange(self.n_class+1)+1)[0]*1.0/self.Y.shape[0]
    def claculate_mean(self):
        #calculate the mean of each class and store them in  self.mean
#        class_data = np.zeros((self.n_class,self.n_feature))q
        for label in np.unique(self.Y):
            self.mean[label-1]=self.X[self.Y[:,0]==label].mean(0)
    def claculate_sigma(self):
        #calculate the covariance of each class and store them in  self.sigma
        sigma_label_ave=np.zeros(shape=(self.n_feature,self.n_feature))
        for label in np.unique(self.Y):
            a=self.X[self.Y[:,0]==label]
            b=a-self.mean[label-1]
            sigma_label=b.transpose().dot(b)
            sigma_label_ave=sigma_label_ave+sigma_label
        sigma_label_ave*1.0/self.X.shape[0]
        self.sigma = np.tile(sigma_label_ave,(self.n_class,1)).reshape(self.n_class,self.n_feature,self.n_feature)
        if(self.is_linear==False):
            for label in np.unique(self.Y):
                a=self.X[self.Y[:,0]==label]
                b=a-self.mean[label-1]
                sigma_label=b.transpose().dot(b)*1.0/a.shape[0]
                self.sigma[label-1] = sigma_label
    def classify(self,x_test):
        y_pre=np.zeros(shape=(x_test.shape[0],self.n_class))
        # after training , use the model to classify x_test, return y_pre
        for label in np.unique(self.Y):
            px_y=density(x_test,self.mean[label -1],self.sigma[label-1])
            y_pre[:,label-1]=px_y*self.pro[label-1]
        y_pre = np.argmax(y_pre,axis=1)+1
        y_pre=y_pre.reshape(-1,1)
        return y_pre

LDA = GDA(x_train,y_train) # generate the LDA model
LDA.calculate_pro()        # calculate parameters
LDA.claculate_mean()
LDA.claculate_sigma()
y_pre = LDA.classify(x_test) # do classify after training
LDA_acc = (y_pre == y_test).mean()
print ('accuracy of LDA is:{:.2f}'.format(LDA_acc))
    

QDA = GDA(x_train,y_train,is_linear=False) # generate the QDA model
QDA.calculate_pro()                     # calculate parameters
QDA.claculate_mean()
QDA.claculate_sigma()
y_pre = QDA.classify(x_test)          # do classify after training
QDA_acc = (y_pre == y_test).mean()
print ('accuracy of QDA is:{:.2f}'.format(QDA_acc))
