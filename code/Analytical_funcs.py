'''Solving GRN steady state analytically'''
from Global_terms import *
#Assume morphogen is at steady state (SS)
params = {'V_max_n1': 30, #production term
          'w_m': 1, #weights
          'w_n2':1, 
          'n_m':1, #hill exponents
          'n_n2':1, 
          'V_max_n2': 30,
          'w_n1': 1.5, 
          'n_n1':1, 
          'Deg_n1': 2, # degradation term 
          'Deg_n2': 2} 


# params_list = list(params.values())
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

# M_SS = M_steady()



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

# N1_SS, N2_SS = GRN_steady(M_SS, par_list=pars_list)