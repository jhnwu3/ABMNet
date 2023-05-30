#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 21:58:00 2022

@author: ixn004
"""

import numpy as np
import bionetgen


def run_bionetgen(n,p):
    
    # n=average over how many bionetgen stochastic pZAP70 trajectories
    # p=pZAP70 in which column number in the bionetgen output file
    
    
    #create n output bionetgen folders
    for  i in range (0,n,1) :
          ret = bionetgen.run("gamma_HPC.bngl", out="HPC_output_"+str(i),suppress=True)
     
    #save the time,pZAP70 data from folder_0
    dir_X= "HPC_output_0"
    X = open(dir_X+'/gamma_HPC.gdat').readlines()   # X is a list with  \t
    l=len(X)
    t=[]
    Y=[] #define a list  to sum over the columns , first define the column of output_0
    #Z=[]
    for j in range (1,l,1): #reading all lines from a folder 
        t.append(float(X[j].strip('\n').split('\t')[0])) #assign the 0th column to time  
        Y.append(float(X[j].strip('\n').split('\t')[p])) #assign the pth column to Y or pZAP70
        #Z.append(float(X[j].strip('\n').split('\t')[p])) #assign the pth column to another observable
        
    
     #averaging between the n folders   
    for i in range (1,n,1) : #reading folders i=1,n
              k=0
              dir_X="HPC_output_"+str(i)
              X = open(dir_X+'/gamma_HPC.gdat').readlines()
              for j in range (1,l,1):
                  Y[k]=Y[k]+float(X[j].strip('\n').split('\t')[p]) #assign the 3rd column to Y
                  t[k]=t[k]+float(X[j].strip('\n').split('\t')[0])
                  k=k+1
    avg=[]
    T=[]
    for j in range (0,l-1,1):
          avg.append(Y[j]/n)
          T.append(t[j]/n)
    return T,avg
     