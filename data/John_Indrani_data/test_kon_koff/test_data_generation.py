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

#modules
import numpy as np
import bionetgen
from run_bngl import run_bionetgen





#module to mean pZAP signal for K1=kon, K2=Koff
def pZAP_signal(K1,K2):
    
    
    #0. Bionetgen parameter set 
    model = bionetgen.bngmodel("gamma_HPC_0.bngl")
    model.parameters.Kab = K1 # assigning new kon
    model.parameters.KU = K2 # assigning new koff
              
   #print(model)

   #print model in directiry name_gamma_HPC.bngl    5 folders
    with open("gamma_HPC.bngl", "w") as f: #write a new file assigning new kon, koff
        f.write(str(model)) # writes the changed model to new_model file
          
          
   # #1. Bionetge run : average PZAP   over n run and obseravble is in the (p+1) column
    n=5 #average over 3  trajectories : John you can change this
    p=1 #2nd column in the file: pZAP is printed on the first column, time is printed on the zeroth column
    T,avg=run_bionetgen(n,p) 
    T=np.asarray(T)
    avg=np.asarray(avg)
    
    return T,avg #avg is is the mean PZAP signal
       


def main():
    
    
        kon=0.001
        koff=0.5
    
        #Generate pZAP signal based on kon,koff
        time,mean_pZAP=pZAP_signal(kon,koff)
      
        f = f"test.txt"  # Generate file name (e.g., output1.txt)
        
        
        with open(f, 'w') as file:
    
            file.write(str(kon)+"\t"+str(koff)+"\n")
            
            for k in range(len(time)):
                file.write(str(time[k])+"\t"+str(mean_pZAP[k])+"\n")
            
        
    
if __name__=="__main__":
    main()
    
            