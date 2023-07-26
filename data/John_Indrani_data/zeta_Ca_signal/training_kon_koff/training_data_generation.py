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
import random
import numpy as np
import pandas as pd
import bionetgen
from run_bngl import run_bionetgen
from interpolation import spline
from ca_ODE import calcium, solve
import matplotlib.pyplot as plt



#module to generate (kon,koff,C1,C2,g) as uniform random numbers
def param_generate(p):
  
 #Generate 5 parameters kon,koff, C1, C2, g
 # kon,koff in bionetgen code
 # C1,C2,g in the Ca ODE equation
 
    #kon range[1E-7,1E-3]
    #koff range [1,10]
    #C1 range [10000,5000]]
    #C2 range [1,10]
    #g range [1E-4, 1E-2]
    
    param=[]
    # indrani_estimates = np.loadtxt("CD3z_46L_indrani_estimates.dat")
    # p = indrani_estimates.shape[0]
    # indrani_estimates = np.power(10, indrani_estimates)
    # load indrani estimates
    kon = 0
    koff = 0
    for i in range(p):
    
        # 350k
        # kon = 3.50749103e-04
        # koff = 3.81711042e+00
        # C1 = 6.22638543e+03
        # C2 = 2.45842800e+00
        # g = 1.74688920e-03
        
        # 264k
        # kon = 1.69782590e-04 
        # koff = 1.18486403e+00
        # C1 = 5.69739756e+03 
        # C2= 3.67541211e+00
        # g = 1.12731354e-03
        
        # 82k
        # kon = 3.07598542e-03 
        # koff = 5.00039210e+00 
        # C1 = 5.07948984e+03 
        # C2= 3.87763875e+00
        # g = 7.67350887e-03
        
        # 525k
        # kon = 1.65079865e-03
        # koff = 5.14053840e+00 
        # C1 = 8.20097588e+03 
        # C2 = 2.91216075e+00
        # g = 3.56477064e-03
        
        # 92k without "full" model
        # kon = 4.59461120e-04 
        # koff = 8.01328689e+00 
        # C1= 8.15367177e+03
        # C2 = 2.86002175e+00
        # g = 2.47548736e-03
        
        # # 22k without full model
        # kon = 9.10917059e-03 
        # koff = 4.20443821e+00
        # C1 = 6.40326243e+03 
        # C2 = 1.60359680e+00
        # g = 5.49390322e-03
        
      
        # kon = 0.00026 
        # koff = 9.23214
        C1 =  4499.434614189878
        C2 = 1.2887
        g = 0.003268352949
        
        
        kon += 0.001 * (i / float(p))
        koff += 10 * (i / float(p))
        # kon += 0.5 * random.uniform(-1e-5, 1e-5) # offset by some amount
        # koff += 0.5 * random.uniform(-1,1)
        # C1 += 0.5 * random.uniform(-100, 100)
        # C2 += 0.5 * random.uniform(-0.1, 0.1)
        # g += 0.5 * random.uniform(-1e-4, 1e-4)
        
        # kon=random.uniform(1e-7,1e-2)
        # koff=random.uniform(1,10)
        # C1=random.uniform(5e3,1e4)
        # C2=random.uniform(1,10)
        # g=random.uniform(1e-4,1e-2)
        
        
        
        # kon = 3.58220915e-03 
        # koff = 6.16259226e+00
        # C1 = 9.36331580e+03 
        # C2 =1.66294341e+00
        # g = 6.80434035e-03
        
        # when standardizing with the standard space sanity check model estimates
        # kon = 9.69399302e-03 
        # koff = 8.52310271e+00 
        # C1 = 9.42838024e+03 
        # C2 = 9.25670780e+00
        # g = 4.31914380e-04
        
        # when standardizing with the sanity check data and keeping sampling bounded within
        
        # kon = 2.6000000e-05 
        # koff = 9.2321500e+00 
        # C1 = 4.4994346e+03 
        # C2 = 1.2887300e+00
        # g = 3.2684000e-03
        
        param.append((kon,koff,C1,C2,g))
        # param.append(indrani_estimates[i,:-1])
        
    return param


#module to mean pZAP signal for K1=kon, K2=Koff
def pZAP_signal(K1,K2):
    
    
    #0. Bionetgen parameter set 
    model = bionetgen.bngmodel("zeta_HPC_0.bngl")
    model.parameters.Kab = K1 # assigning new kon
    model.parameters.KU = K2 # assigning new koff
              
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
   
    #*****************John you can change
    p=1000 #the number of the parameter sets of [kon,koff,C1,C2,g]
    
    
    #actual experimental data
    data=pd.read_csv('oscar_ca.dat',sep="\t", comment='#', header=None)
    global t,exp_data
    data=np.asarray(data) 
    t=data[:,0]
    exp_data=data[:,1]
    
   
    N=2000 #Ca signal at N time points 
    tstart=25.0 # Fit to be start from which timepoint : interpolation of pZAP70 signal starts at 25 sec
    Vc=25 #pZAP molecules in the simulation box of size 25 um^3
    z=602 #constant factor to convert from molecules/um3 to uM
    
    param=param_generate(p)
    param=np.asarray(param) #array of kon,koff
    
    #Number of (kon,koff) exists, printing in the files
    for i in range(len(param)) :
        f = f"../contour_line/train{i}_ca_h.txt"  # Generate file name (e.g., output1.txt)
        
        #load each set of (kon, koff,C1,C2,g)
        kon=param[i,0]
        koff=param[i,1]
        C1=param[i,2]
        C2=param[i,3]
        g=param[i,4]
    
        #1.Generate pZAP signal based on kon,koff
        time,mean_pZAP=pZAP_signal(kon,koff)
        
        #2.Interpolate time, PZAP  from 600 points to 2000 points   
        tnew,PZAP=spline(time, mean_pZAP, N,tstart)
        tnew=np.asarray(tnew)
        PZAP=np.asarray(PZAP) #total number of PZAP molecule in the simulation box of size Vc
        #plt.plot(time,mean_pZAP,'co',tnew,PZAP,'k-')
        PZAP=PZAP/(Vc*z) #pZAP in uM unit
        
        
        
        #3. Ca Signal  from the ODE model feeding the PZAP signal   
        CA0=33212 #do not change
        y0 = [CA0, 10]
        ca,h=solve(tnew,N,y0,PZAP,C1,C2,g)
        ca=np.asarray(ca)
        h=np.asarray(h)
       
    
        #plt.plot(tnew,PZAP)
        
        #plt.plot(t,exp_data,'b-',tnew,ca,'k')
    
        
        #print tnew, PZAP, ca in the files
        with open(f, 'w') as file:
    
            file.write(str(kon)+"\t"+str(koff)+"\t"+str(C1)+"\t"+str(C2)+"\t"+str(g)+"\n")
            
            for k in range(len(tnew)):
                file.write(str(tnew[k])+"\t"+str(PZAP[k])+"\t"+str(ca[k])+"\t" +str(h[k]) +"\n")
            
      
    
if __name__=="__main__":
    main()
    
            