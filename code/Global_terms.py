import numpy as np
#%%
'''Global terms'''
L = 2.0 #width of tissue, size of cell 50 
K = 4.8 #linear degradation, for a decay length of 0.25 (4.8); 0.5 (1.2)
D = 0.3 #Diffusion coefficient - Bicoid micrometer^2/s
v = 100 #production rate
N = 501 - 1
dx = L/N
decay_length = np.sqrt((D/K))
grid = N + 1 #includes 0 and L
source_bound = 0.2 * L/dx #10% of the tissue width
x = np.linspace(0,L,grid) #positions i.e number of cells between the width of the tissue
N_t = 100

#timesteps
dt = (dx**2)/(2*D) *0.75  #must be small enough to create a stable solution.
t = 0
tmax = 10/K #one order of magnitude greater than 1/K
#%%