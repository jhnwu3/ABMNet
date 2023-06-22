#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 14:06:13 2023

@author: ixn004
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 22:04:27 2022

@author: ixn004
"""
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d


def spline (T,avg,N,tstart):
    f=sp.interpolate.interp1d(T,avg,kind='cubic')
    tnew= np.linspace(tstart, max(T), num=N, endpoint=True)
    PZAP=f(tnew) #type
    return tnew,PZAP