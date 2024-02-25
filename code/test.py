#%%
from Global_terms import *
import math
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from tqdm import tqdm

#%%
def M_analytical():
    ''' solution for the steady state of a given specific parameters for a morphogen gradient'''
    M_SS = []
    for i in range(0, int(source_bound+1)): #analytical solution taken 
        #included x=0 to x=source bound as production terms
        # M = (v/K) * (1 - ((math.sinh((L/decay_length)-((source_bound*dx)/decay_length)) / math.sinh(L/decay_length)) * math.cosh((i*dx)/decay_length))) 

        M = (v/K) * (1-((np.sinh((L/decay_length) - ((source_bound*dx)/decay_length)) / (np.sinh(L/decay_length))) * np.cosh((i*dx)/decay_length)))

        M_SS.append(M)

    for i in range(int(source_bound+1), len(x)):
        # M = (v/K) * ((math.sinh(((source_bound*dx)/decay_length)) / math.sinh(L/decay_length)) *math.cosh((L-(i*dx))/decay_length))

        M = (v/K) * ((np.sinh((source_bound*dx)/decay_length) / np.sinh(L/decay_length)) * np.cosh((L - (i*dx)) /decay_length))

        M_SS.append(M)

    return np.array(M_SS)

M_ana = M_analytical()

def M_computational():
    d = D/(dx**2)
    length = len(x)
    Nm = N-1
    M0 = np.zeros(length)
    dCdt = np.zeros(length)  
    t_passed = 0
    t_passed_all = 0
    def PDE_M(c, t): 
        """ PDE solver, requires C(x) for a given t """
        dCdt[0] = d * (2*(c[1] - c[0])) - K*c[0] + v

        for i in range(1,int(source_bound+1)):
            dCdt[i] = d * (c[i+1] - 2*c[i] + c[i-1]) - K*c[i] + v
        for i in range(int(source_bound+1), N):
            dCdt[i] = d * (c[i+1] - 2*c[i] + c[i-1]) - K*c[i]
        
        dCdt[N] = d * (2*(c[Nm] - c[N])) - K*c[N]
        # dCdt[int(length)-1] = d * (2*(c[int(length)-2] - c[int(length)-1])) - K*c[int(length)-1]

        return c + dCdt*dt
    
    # def PDE_M(c, t): 
    #     """ PDE solver, requires C(x) for a given t """
    #     dCdt[0] = d * (2*(c[1] - c[0])) - K*c[0] + v

    #     for i in range(1,100):
    #         dCdt[i] = d * (c[i+1] - 2*c[i] + c[i-1]) - K*c[i] + v
    #     for i in range(100, 500):
    #         dCdt[i] = d * (c[i+1] - 2*c[i] + c[i-1]) - K*c[i]
        
    #     dCdt[500] = d * (2*(c[499] - c[500])) - K*c[500]
    #     # dCdt[int(length)-1] = d * (2*(c[int(length)-2] - c[int(length)-1])) - K*c[int(length)-1]

    #     return c + dCdt*dt

    Mt = PDE_M(M0, t)
    
    while t_passed <= 15/K:
        Mt = PDE_M(Mt,t)
        t_passed += dt
    return Mt

M_comp = M_computational()
#1 is N, 2 is 0, 3 is -1

# %%
temp = np.linspace(0,L,int(L/dx)+1)/L
plt.plot(temp, M_ana, c = 'gold', label = 'M analytic', linestyle = 'dashed')
plt.plot(temp, M_comp, c = 'darkorange', label = 'M simulation')
plt.legend()
plt.xlabel('x/L')
plt.ylabel('conc')
plt.title('Comparison between simulated and analytical SS solutions(Concentration profiles)')
plt.show()

#visualising the difference across
Mdiff = []
for Mc, Ma in zip(M_comp, M_ana):
    Mdiff.append(((Ma - Mc)/Ma)*100) #percentage difference

plt.plot(temp, Mdiff, c = 'darkorange', label= 'M')
plt.legend()
plt.xlabel('x/L')
plt.ylabel('Percentage difference from analytical(%)') #do i need to log you? relative difference and that
plt.title('Comparison between simulated and analytical SS solutions (Quantifying difference)')
plt.show()

# %%
