'''Simulation of morphogen diffusion dynamics and single-cell GRN response'''
#%%
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import math
from scipy.interpolate import interp1d
from Global_terms import *
from GRNs import *
from typing import Any, Dict, List, Optional, Union
from Equations import *
from Analytical_funcs import *
#%%
'''Simulating morphogen concentration profiles across time and cells'''

#start with concentration of morphogen at each cell at time = 0.
#Discritization of PDE
d = D/(dx**2)
length = len(x)
Nm = N - 1
dCdt = np.zeros(length)  
t_passed = 0
sb = int(source_bound+1)
lm = int(length-1)
def PDE_M(c): 
    """ PDE solver, requires C(x) for a given t """
    dCdt[0] = d * (2*(c[1] - c[0])) - K*c[0] + v

    for i in range(1,int(source_bound+1)):
        dCdt[i] = d * (c[i+1] - 2*c[i] + c[i-1]) - K*c[i] + v
    for i in range(int(source_bound+1), N):
        dCdt[i] = d * (c[i+1] - 2*c[i] + c[i-1]) - K*c[i]
    
    dCdt[N] = d * (2*(c[Nm] - c[N])) - K*c[N]
    # dCdt[int(length)-1] = d * (2*(c[int(length)-2] - c[int(length)-1])) - K*c[int(length)-1]

    return c + dCdt*dt

#Simulation of morphogen gradient over time
def get_conc_profile(prior_timestep):
    C0 = np.zeros(length) #all cells have zero morphogen.
    Ct = PDE_M(C0)
    C_log = pd.DataFrame(Ct) #Storage of concentration profiles
    C_log = C_log.transpose()

    t_passed = 0
    t_passed_all = 0
    threshold = 10e-8
    Steady = False
    
    if prior_timestep == None:
        count = 0
        C_log.to_csv('../data/Concentration_profiles_fixed.csv', index=False)
        path = '../data/Concentration_profiles_fixed.csv'
    else:
        count = prior_timestep
        path = '../data/Concentration_profiles_fixed.csv'
        C_log = pd.read_csv(path)
        Ct = C_log.loc[C_log.index[-1]]
    SS_time = 15/K
    bar = pb.ProgressBar(maxval = SS_time+tmax/(N_t)*1.1).start()

    while t_passed_all <= 5*tmax/(N_t):
        Ct = PDE_M(Ct)
        t_passed += dt
        if t_passed >= tmax/(N_t): #Checks every time step
            count += 1
            C_log.loc[len(C_log)] = Ct #stores list of concentration into array
            # print('timestep ', count, 'completed')
            C_temp = pd.DataFrame(Ct)
            C_temp = C_temp.transpose()
            C_temp.to_csv(path,mode='a', header=False, index=False)
        t_passed_all += t_passed
        bar.update(t_passed_all)


    while Steady == False:
        Ct = PDE_M(Ct)
        t_passed += dt
        if t_passed >= tmax/(N_t): #Checks every time step
            count += 1
            C_log.loc[len(C_log)] = Ct #stores list of concentration into array
            # print('timestep ', count, 'completed')
            C_temp = pd.DataFrame(Ct)
            C_temp = C_temp.transpose()
            C_temp.to_csv(path,mode='a', header=False, index=False)

            # C_diff = [abs((a - b)/a) for a, b in zip(C_log.loc[C_log.index[-2]], C_log.loc[C_log.index[-1]])]
        
            # if max(C_diff) <= threshold:
            #     Steady = True
            t_passed_all += t_passed
            if t_passed_all >= SS_time:
                Steady = True
            t_passed = 0
            bar.update(t_passed_all)


        
###savefig
#get_conc_profile(prior_timestep=None) #3mins
C_log = pd.read_csv('../data/Concentration_profiles_fixed.csv')
C0 = C_log.loc[1]
x_L = np.arange(0,N+1)
plt.plot(x, C0, c = 'orange', linestyle = 'dashed', label = 'Growing Morphogen')
for i in range(0,C_log.shape[0],8):
    C = C_log.loc[i].to_list()
    plt.plot(x, C,c = 'orange', linestyle = 'dashed')
plt.plot(x, C_log.loc[C_log.shape[0]-1], c= 'orange', label = 'Steady state')
plt.legend(bbox_to_anchor=(1.1, 1.05))
plt.xlabel('x')
plt.ylabel('[Morphogen]')
plt.show()
#plt.savefig('../results/Morphogen_dynamics.pdf')


# %%
'''Regulatory Network, Set or change params below and computationally solve the solution.'''
params = {'V_max_n1': 30, #production term
          'w_m': 1, #weights
          'w_n2':1, 
          'n_m':1, #hill exponents
          'n_n2':1, 
          'V_max_n2': 30,
          'w_n1': 1, 
          'n_n1':1, 
          'Deg_n1': 2, # degradation term 
          'Deg_n2': 2} 


pars_list = list(params.values())

#initialisation
GRN = test_model(pars_list, pd.DataFrame(C0).transpose()) 

####Computational simulation of the dynamic system from initial conditions zero
N1_conc, N2_conc, M_conc = GRN.simple_dynamical(pars_list, dt, N) #simple_dynamical for growing curve
#Visualising the change in each species over time (each frame is a seperate time step).
# for j in range(0,M_conc.shape[0],3):
#     plt.plot(x,N1_conc.loc[j], c = 'teal',label='node 1')
#     plt.plot(x,N2_conc.loc[j], c = 'grey', label= 'node 2')
#     plt.plot(x,M_conc.loc[j], c='darkorange', label = 'Morphogen')
#     plt.legend(bbox_to_anchor=(1.1, 1.05))
#     plt.xlabel('L')
#     plt.ylabel('conc')
#     plt.show()


#Visualising the change in each species over time (All timesteps are combined into one)
for j in range(0, M_conc.shape[0]-1, 5000):
    plt.plot(x,N1_conc.loc[j], c = 'teal',linestyle = 'dashed')
    plt.plot(x,N2_conc.loc[j], c = 'grey', linestyle = 'dashed')
    plt.plot(x,M_conc.loc[j], c='darkorange', linestyle = 'dashed')
    plt.xlabel('L')
    plt.ylabel('conc')

plt.plot(x,N1_conc.loc[M_conc.index[-1]], c = 'teal',label='node 1')
plt.plot(x,N2_conc.loc[M_conc.index[-1]], c = 'grey', label= 'node 2')
plt.plot(x,M_conc.loc[M_conc.index[-1]], c='darkorange', label = 'Morphogen')
plt.legend(bbox_to_anchor=(1.1, 1.05))
plt.show()
#plt.savefig('../results/GRN_dynamics_GSS.pdf')

#%%
'''Solving GRN steady state analytically'''
#Assume morphogen is at steady state (SS)
# def M_steady():
#     M_SS = []
#     for i in range(0, int(source_bound+1)): #analytical solution taken 
#         #included x=0 to x=source bound as production terms
#         # M = (v/K) * (1 - ((math.sinh((L/decay_length)-((source_bound*dx)/decay_length)) / math.sinh(L/decay_length)) * math.cosh((i*dx)/decay_length))) 

#         M = (v/K) * (1-((np.sinh((L/decay_length) - ((source_bound*dx)/decay_length)) / (np.sinh(L/decay_length))) * np.cosh((i*dx)/decay_length)))

#         M_SS.append(M)

#     for i in range(int(source_bound+1), int(L/dx)):
#         # M = (v/K) * ((math.sinh(((source_bound*dx)/decay_length)) / math.sinh(L/decay_length)) *math.cosh((L-(i*dx))/decay_length))

#         M = (v/K) * ((np.sinh((source_bound*dx)/decay_length) / np.sinh(L/decay_length)) * np.cosh((L - (i*dx)) /decay_length))

#         M_SS.append(M)

#     return np.array(M_SS)

M_SS = M_steady()



# def GRN_steady(M_SS, par_list):
#     '''Analytical solution to GRN for hill exponent of 1'''
#     params_list = par_list
#     V_max_n1 = params_list[0]
#     w_m = params_list[1] 
#     w_n2 = params_list[2]
#     n_m = params_list[3]
#     n_n2 = params_list[4]

#     V_max_n2 = params_list[5]
#     w_n1 = params_list[6]
#     n_n1 = params_list[7]

#     Deg_n1 = params_list[8] #Assumes linear degradation 
#     Deg_n2 = params_list[9]

#     a = (Deg_n1*w_n1)*(1+w_m*M_SS) 
#     b = Deg_n1*(1+(w_m*M_SS)+((w_n2*V_max_n2)/Deg_n2)-((V_max_n1*w_m*M_SS*w_n1)/Deg_n1))
#     c = (V_max_n1*w_m*M_SS)

#     #completing the square
    
#     N1_SS = np.sqrt(c/a + ((b/a)/2)**2) - (b/a)/2

#     N2_SS = (V_max_n2/(1+(w_n1*N1_SS)))/Deg_n2

#     return N1_SS, N2_SS

N1_SS, N2_SS = GRN_steady(M_SS, par_list=pars_list)

## Checking if all values of X for the simulations match up to the analytical solution. 

N1_conc_SS = N1_conc.loc[N1_conc.index[-1]] #Take steady state concentrations only
N2_conc_SS = N2_conc.loc[N2_conc.index[-1]]
M_conc_SS = M_conc.loc[M_conc.index[-1]]

# N1_conc_SS = N1_conc
# N2_conc_SS = N2_conc
# M_conc_SS = M_conc

plt.plot(x/L, N1_conc_SS, c = 'teal', label = 'N1 simulation')
plt.plot(x/L, N1_SS, c = 'teal', label = 'N1 analytic', linestyle = 'dashed')
plt.plot(x/L, N2_conc_SS, c = 'grey', label = 'N2 simulation')
plt.plot(x/L, N2_SS, c = 'grey', label = 'N2 analytic', linestyle = 'dashed')
plt.plot(x/L, M_conc_SS, c = 'darkorange', label = 'M simulation')
plt.plot(x/L, M_SS, c = 'gold', label = 'M analytic', linestyle = 'dashed')
plt.legend(bbox_to_anchor=(1.2, 1)) 
plt.xlabel('x')
plt.ylabel('concentration')
plt.title('Comparison between simulated and analytical SS solutions')
plt.show()

#visualising the difference across
N1diff, N2diff, Mdiff = [], [], []
for N1s, N2s, Ms, N1a, N2a, Ma in zip(N1_conc_SS,N2_conc_SS,M_conc_SS,N1_SS,N2_SS,M_SS):
    N1diff.append((N1a - N1s)/N1a*100)
    N2diff.append((N2a - N2s)/N2a*100)
    Mdiff.append((Ma - Ms)/Ma*100)

plt.plot(x/L, N1diff, c = 'teal', label= 'N1')
plt.plot(x/L, N2diff, c = 'grey', label= 'N2')
plt.plot(x/L, Mdiff, c = 'darkorange', label= 'M')
plt.legend(bbox_to_anchor=(1.2, 1))
plt.xlabel('x/L')
plt.ylabel('Perecentage difference to analytical (%)') #do i need to log you? relative difference and that
plt.title('Comparison between simulated and analytical SS solutions (Quantifying difference)')
plt.show()

#looking at difference in a nother way
# plt.scatter(N1_SS,N1_conc_SS, c = 'teal', alpha = 0.6)
# a = np.arange(np.min(N1_SS),np.max(N1_SS),1)
# plt.plot(a,a, c = 'black', linestyle='dashed')
# plt.ylabel('Simulation')
# plt.xlabel('Analytical')
# plt.title('N1 comparison')
# plt.show()
# plt.scatter(M_SS,M_conc_SS, c = 'darkorange', alpha = 0.6)
# a = np.arange(np.min(M_SS),np.max(M_SS),1)
# plt.plot(a,a, c = 'black', linestyle='dashed')
# plt.ylabel('Simulation')
# plt.xlabel('Analytical')
# plt.title('M comparison')
# plt.show()
# plt.scatter(N2_SS,N2_conc_SS, c = 'grey', alpha = 0.6)
# a = np.arange(np.min(N2_SS),np.max(N2_SS),1)
# plt.plot(a,a, c = 'black', linestyle='dashed')
# plt.ylabel('Simulation')
# plt.xlabel('Analytical')
# plt.title('N2 comparison')
# plt.show()


#code for a function which takes a csv of timesteps and can run the simulations from there so I don't have to run it from scratch everytime for the same set of parameters.




#%%

# %%
'''Investigating boundary dynamics'''
X_thr_n1, X_thr_n2, M_thr_n1, M_thr_n2 = [], [], [], []

for N1, N2, M in zip(N1_conc.index,N2_conc.index,M_conc.index):
    N1_thr = N1_conc.loc[N1].max()*0.95 #Threshold of boundary condition specified as a 5% range from the max.
    N1_list = N1_conc.loc[N1].to_list()

    #interpolate between cell boundaries to find the exact position of X_thr?
    interp_func = interp1d(N1_list,x)
    X_thr = interp_func(N1_thr)
    X_thr_n1.append(X_thr)

    M_list = M_conc.loc[M].to_list()

    M_interp_func = interp1d(x,M_list)
    M_thr = M_interp_func(X_thr)
    M_thr_n1.append(M_thr)

    N2_thr = N2_conc.loc[N2].max()*0.95 
    N2_list = N2_conc.loc[N2].to_list()

    interp_func = interp1d(N2_list,x)
    if N2_thr < N2_list[0]:
        N2_thr = N2_list[0]
    X_thr = interp_func(N2_thr)
    X_thr_n2.append(X_thr)

    M_interp_func = interp1d(x,M_list)
    M_thr = M_interp_func(X_thr)
    M_thr_n2.append(M_thr)

    check = 0
    i = 0

#Plotting node boundary dynamics over time(growth of morpho gradient)
timestep = np.arange(0,M_conc.index[-1]+1)
timestep = timestep*(tmax/N_t) #Absolute time

#N1
X_thr_n1[0] = X_thr_n1[1] #first condition is assuming morphogen gradient is 0, replaced with initial conditions
plt.scatter(timestep, X_thr_n1, c = 'teal', label='N1')
X_thr_n2[0] = X_thr_n2[1]
plt.scatter(timestep, X_thr_n2, c = 'grey', label='N2')
plt.title('GRN Spatial boundary (X) dynamics across time')
plt.xlabel('Time (s)')
plt.ylabel('X_i')
plt.legend(bbox_to_anchor=(1.2, 1))
plt.show()

#Plotting node boundary dynamics with morphogen concnentration
plt.scatter(M_thr_n1, X_thr_n1, c = 'teal',label='N1')
plt.scatter(M_thr_n2, X_thr_n2, c = 'grey',label='N1')
plt.title('GRN spatial boundary(X) position and morphogen concentration dynamics')
plt.xlabel('Morphogen concentration')
plt.ylabel('X_i')
plt.legend(bbox_to_anchor=(1.2, 1))
plt.show() 

# %%
'''Checking if running GRN from 0 and morphogen at SS makes a difference to final solution'''
M = M_steady()
# M = np.array(M_conc.loc[M_conc.index[-1]])
N1_conc_2, N2_conc_2, M_conc_2 = GRN.simple_ss_m(pars_list, M, dt, N)

plt.plot(x/L, N1_SS, c = 'teal', label = 'N1 analytic', linestyle = 'dashed')
plt.plot(x/L, N1_conc_2.loc[N1_conc_2.index[-1]], c = 'teal', label = 'N1 simulation')
plt.plot(x/L, N2_SS, c = 'grey', label = 'N2 analytic', linestyle = 'dashed')
plt.plot(x/L, N2_conc_2.loc[N2_conc_2.index[-1]], c = 'grey', label = 'N2 simulation')
plt.plot(x/L, M_SS, c = 'gold', label = 'M analytic', linestyle = 'dashed')
plt.plot(x/L, M_conc_2.loc[M_conc_2.index[-1]], c = 'darkorange', label = 'M simulation')
plt.legend(bbox_to_anchor=(1.1, 1.05))
plt.xlabel('x/L')
plt.ylabel('conc')
plt.title('Comparison between simulated and analytical SS solutions(Concentration profiles)')
plt.show()

N1diff, N2diff, Mdiff = [], [], []
for N1s, N2s, Ms, N1a, N2a, Ma in zip(N1_conc_2.loc[N1_conc_2.index[-1]], N2_conc_2.loc[N2_conc_2.index[-1]],M_conc_2.loc[M_conc_2.index[-1]],N1_SS,N2_SS,M_SS):
    N1diff.append((N1a - N1s)/N1a*100)
    N2diff.append((N2a - N2s)/N2a*100)
    Mdiff.append((Ma - Ms)/Ma*100)

plt.plot(x/L, N1diff, c = 'teal', label= 'N1')
plt.plot(x/L, N2diff, c = 'grey', label= 'N2')
plt.plot(x/L, Mdiff, c = 'darkorange', label= 'M')
plt.legend()
plt.xlabel('x/L')
plt.ylabel('percentage difference from analytical solution') #do i need to log you? relative difference and that
plt.title('Comparison between simulated and analytical SS solutions (Quantifying difference)')
plt.show()
# %%
'''Checking if fraction between boundary conditions is constant for different amplitudes of the morphogen'''

#timesteps
dt = (dx**2)/(2*D) *0.75  #must be small enough to create a stable solution.
t = 0
tmax = 10/K #one order of magnitude greater than 1/K

decay_length = np.sqrt((D/K))
params = {'V_max_n1': 30, #production term
          'w_m': 3, #weights
          'w_n2':1.5, 
          'n_m':1, #hill exponents
          'n_n2':1, 
          'V_max_n2': 30,
          'w_n1': 1.5, 
          'n_n1':1, 
          'Deg_n1': 2, # degradation term 
          'Deg_n2': 2} 


pars_list = list(params.values())

K_list = [4.8,2.1333,1.2]
Amp_list = [50,100,200]
frac_list = []
decay_length_list = []

for k in K_list:
    K = k
    L = 2.0 #width of tissue, size of cell 50 
    #linear degradation, for a decay length of 0.25 (4.8); 0.5 (1.2)
    D = 0.3 #Diffusion coefficient - Bicoid micrometer^2/s
    v = 100 #production rate
    N = 501 - 1
    dx = L/N
    decay_length = np.sqrt((D/K))
    grid = N + 1 #includes 0 and L
    source_bound = 0.2 * L/dx #10% of the tissue width
    x = np.linspace(0,L,grid) #positions i.e number of cells between the width of the tissue
    N_t = 100

    #initialisation
    GRN = test_model(pars_list, pd.DataFrame(C0).transpose())
    def M_steady():
        M_SS = []
        for i in range(0, int(source_bound+1)): #defining break at exactly the source bound will lead to error as its discontinous
            #included x=0 to x=source bound as production terms
            # M = (v/K) * (1 - ((math.sinh((L/decay_length)-((source_bound*dx)/decay_length)) / math.sinh(L/decay_length)) * math.cosh((i*dx)/decay_length))) 

            M = (v/K) * (1-((np.sinh((L/decay_length) - ((source_bound*dx)/decay_length)) / (np.sinh(L/decay_length))) * np.cosh((i*dx)/decay_length)))

            M_SS.append(M)

        for i in range(int(source_bound+1), len(x)):
            # M = (v/K) * ((math.sinh(((source_bound*dx)/decay_length)) / math.sinh(L/decay_length)) *math.cosh((L-(i*dx))/decay_length))

            M = (v/K) * ((np.sinh((source_bound*dx)/decay_length) / np.sinh(L/decay_length)) * np.cosh((L - (i*dx)) / decay_length))

            M_SS.append(M)

        return np.array(M_SS)
    M_SS = M_steady()
    def GRN_steady(M_SS, par_list):
        '''Analytical solution to GRN for hill exponent of 1'''
        params_list = par_list
        V_max_n1 = params_list[0]
        w_m = params_list[1] 
        w_n2 = params_list[2]
        n_m = params_list[3]
        n_n2 = params_list[4]

        V_max_n2 = params_list[5]
        w_n1 = params_list[6]
        n_n1 = params_list[7]

        Deg_n1 = params_list[8] #Assumes linear degradation 
        Deg_n2 = params_list[9]

        a = (Deg_n1*w_n1)*(1+w_m*M_SS) 
        b = Deg_n1*(1+(w_m*M_SS)+((w_n2*V_max_n2)/Deg_n2)-((V_max_n1*w_m*M_SS*w_n1)/Deg_n1))
        c = (V_max_n1*w_m*M_SS)

        #completing the square
        
        N1_SS = np.sqrt(c/a + ((b/a)/2)**2) - (b/a)/2

        N2_SS = (V_max_n2/(1+(w_n1*N1_SS)))/Deg_n2

        return N1_SS, N2_SS
    N1_SS, N2_SS = GRN_steady(M_SS, par_list=pars_list)

    # plt.plot(M_SS, c = 'darkorange')
    # plt.plot(N1_SS, c = 'teal')
    # plt.plot(N2_SS, c = 'grey')

    N1_thr = max(N1_SS)*0.95 #Threshold of boundary condition specified as a 5% range from the max.
    #interpolate between cell boundaries to find the exact position of X_thr?
    interp_func = interp1d(N1_SS,x)
    X1_thr = interp_func(N1_thr)/L

    N2_thr = max(N2_SS)*0.95 #Threshold of boundary condition specified as a 5% range from the max.
    #interpolate between cell boundaries to find the exact position of X_thr?
    interp_func = interp1d(N2_SS,x)
    X2_thr = interp_func(N2_thr)/L

    frac = X1_thr/X2_thr

    frac_list.append(frac)
    decay_length_list.append(decay_length)

plt.plot(decay_length_list,frac_list)
plt.ylabel('X1/L / X2/L')
plt.xlabel('decay length')



# %%
