#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 17:00:28 2023

@author: ixn004
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 12:17:44 2023

@author: ixn004
"""
#zeta code

#modules
import numpy as np
import bionetgen
from run_bngl import run_bionetgen


#module to generate (kon,koff) uniformly spaced 
def param_generate(x_points,y_points,x_gap,y_gap):
    param=[]
    for i in range(x_points):
        for j in range(y_points):
            param.append((i*x_gap,j*y_gap))
    return param


#module to mean pZAP signal for K1=kon, K2=Koff
def pZAP_signal(K1,K2):
    
    
    #0. Bionetgen parameter set 
    model = bionetgen.bngmodel("zeta_HPC_0.bngl")
    model.parameters.Kab = K1 # assigning new kAB
    model.parameters.KU = K2 # assigning new kU
              
   #print(model)

   #print model in directiry name_gamma_HPC.bngl    5 folders
    with open("zeta_HPC.bngl", "w") as f: #write a new file assigning new kon, koff
        f.write(str(model)) # writes the changed model to new_model file
          
          
   # #1. Bionetge run : average PZAP   over n run and obseravble is in the (p+1) column
    n=5 #average over 3  trajectories : John you can change this
    p=1 #2nd column in the file: pZAP is printed on the first column, time is printed on the zeroth column
    T,avg=run_bionetgen(n,p) 
    T=np.asarray(T)
    avg=np.asarray(avg)
    
    return T,avg #avg is is the mean PZAP signal
       


def main():
    
    
    #**************John you can change these
    x_points=50 # number of x points
    y_points=50 #number of y points
    
 
    xrange=(10**(-3)-10**(-7)) #(xmax - xmin)
    yrange=(10.0-1.0) # (ymax-ymin)
    x_gap=(xrange/x_points) #deltay for parameter sweep kAB
    y_gap=(yrange/y_points) #deltax for parameter sweep kU
    
    #This will generate the parameter sets for the kon and koff
    param=param_generate(x_points,y_points,x_gap,y_gap)
    param=np.asarray(param) #array of kon,koff
    print(param)
    #***************************************************
    
    #Number of (kon,koff) exists, printing in the files
    for i in range(len(param)) :
        f = f"train{i}.txt"  # Generate file name (e.g., output1.txt)
        
        #each set of (kon, koff)
        kon=param[i,0]
        koff=param[i,1]
    
        #Generate pZAP signal based on kon,koff
        time,mean_pZAP=pZAP_signal(kon,koff)
      
    
        
        with open(f, 'w') as file:
    
            file.write(str(kon)+"\t"+str(koff)+"\n")
            
            for k in range(len(time)):
                file.write(str(time[k])+"\t"+str(mean_pZAP[k])+"\n")
            
        
    
if __name__=="__main__":
    main()
    
            