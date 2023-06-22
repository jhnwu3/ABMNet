#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 14:01:09 2023

@author: ixn004
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 22:17:27 2022

@author: ixn004
"""

import numpy as np
from scipy.integrate import odeint

def calcium(y, t, PZAP,C1,C2,g):
  
    
    
    #z=602 #from uM unit to molecules/mu^3 unit
   
    be=0
    b=0.111 # maximum Ca flow from Cyto to ER  
    k1=0.7 # In Atri 1993, has unit of uM 
    k2=0.7 # In Atri 1993, has unit of uM 
    s=(k2*k2) #
    
   
 
    
    ca, h = y
    #pZAP is in the uM unit
    
    #dcadt=(kf*h*v*mu*PZAP*(((b*k1)+ca)/(k1+ca)))-((g*ca)/(kg+ca))+(be)
    dcadt=(C1*h*PZAP*(((b*k1)+ca)/(k1+ca)))-(g*ca)+(be)
    dhdt=(C2*PZAP)*((s/(s+ca ** 2))-h)
    
    # np.asanyarray(Y_smooth,dtype=np.float64)
   
    dydt = [dcadt,dhdt]
    return dydt

def solve(tnew,N,y0,PZAP,C1,C2,g):
# store solution
    #empty_like_sets_array_of_same_shape_and_size
    ca = np.empty_like(tnew)
    h = np.empty_like(tnew)
    # record initial conditions
    ca[0] = y0[0]
    h[0] = y0[1]
    #PZAP=np.ones(len(PZAP))

    # solve ODE
    for i in range(1,N):
        # span for next time step
        tspan = [tnew[i-1],tnew[i]]
        # solve for next step
        sol = odeint(calcium,y0,tspan,args=(PZAP[i],C1,C2,g))
        # store solution for plotting
        ca[i] = sol[1][0]
        h[i] = sol[1][1]
        # next initial condition
        y0 = sol[1]
        
    return ca,h    