import numpy as np
from Global_terms import *
#%%
#DE solvers
#Discritization of PDE
def PDE_M(c, t): 
            """ PDE solver, requires C(x) for a given t """
            dCdt = np.zeros(int(L/dx))
            dCdt[0] = 0 # boundary doesnt change = 0
            for i in range(1,int(L/dx)-1):
                if i < 20:
                    V = 100
                else:
                    V = 0
                dCdt[i] = (D/dx**2)*(c[i+1] - 2*c[i] + c[i-1]) - K*c[i] + V
            dCdt[N] = dCdt[N-1]
            dCdt[0] = dCdt[1]
            return c + dCdt*dt

#Need to parameters as either part of function or define it elsewhere as a standalone.
def ODE_N1(n1,t,conc,N2t):
    dN1dt = V_max_n1 * np.power((w_m*conc),n_m)
    dN1dt /= 1 + np.power((w_m*conc),n_m) + np.power((w_n2*N2t),n_n2) 
    dN1dt -= Deg_n1 * n1
    N1 = n1 + dN1dt * dt 
    return N1 #returns N1 concentration after every dt

def ODE_N2(n2,t,conc,N1t):
    dN2dt = V_max_n2
    dN2dt /= 1 + np.power((w_n1*N1t),n_n1)
    dN2dt -= Deg_n2 * n2
    N2 = n2 + dN2dt*dt
    return N2