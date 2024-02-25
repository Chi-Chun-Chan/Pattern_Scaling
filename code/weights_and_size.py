#%%
from weight_analysis import *
from scipy.optimize import minimize
#%%
'''Global terms'''
L = 1.4 #width of tissue, size of cell 50 
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

#M_SS = M_steady()



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
#%%
'''Exploring how modification of weights (c) can alter boundary dynamics.'''
#Modifiable weight range
weights: List[Dict[str, Union[str, float]]] = [{
    'Species': 'Morphogen',
    'lower_limit': -2.0, #log10 scale
    'upper_limit': 2.0
}, {
    'Species': 'N1',
    'lower_limit': -2.0,
    'upper_limit': 2.0
}, {
    'Species': 'N2',
    'lower_limit': -2.0,
    'upper_limit': 2.0
}] 

def weight_dynamics(species,dw):
    '''Look at the change in boundary conditions for different weight values for a given species'''
    increment = dw #increments of weights to be tested for a given range.

    all_weights = []
    for weight in weights:
        if weight['Species'] == species:
            weight_list = np.arange(weight['lower_limit'],weight['upper_limit'],increment)
            all_weights.append(weight_list)

    #Look at different weights for one species at a time.
    count = 0
    X1_thr_list = []
    X2_thr_list = []
    M1_thr_list = []
    M2_thr_list = []
    for w in all_weights[0]:
        if species == 'Morphogen':
            params = {'V_max_n1': 30, 
                    'w_m': 10**w, 
                    'w_n2':1, 
                    'n_m': 1, 
                    'n_n2': 1, 
                    'V_max_n2': 30,
                    'w_n1': 1, 
                    'n_n1': 1, 
                    'Deg_n1': 2, 
                    'Deg_n2': 2}
            w_label = 'w_m'
        elif species == 'N1':
            params = {'V_max_n1': 30, 
                    'w_m': 1, 
                    'w_n2':1, 
                    'n_m': 1, 
                    'n_n2': 1, 
                    'V_max_n2': 30,
                    'w_n1': 10**w, 
                    'n_n1': 1, 
                    'Deg_n1': 2, 
                    'Deg_n2': 2}
            w_label = 'w_n1'
        else:
            params = {'V_max_n1': 30, 
                    'w_m': 1, 
                    'w_n2':10**w, 
                    'n_m': 1, 
                    'n_n2': 1, 
                    'V_max_n2': 30,
                    'w_n1': 1, 
                    'n_n1': 1, 
                    'Deg_n1': 2, 
                    'Deg_n2': 2}
            w_label = 'w_n2'
            
        pars_list = list(params.values())

        #for M weights I can just simulate morphogen gradient separatly and then simulate GRN
        #for N1/N2 weights M SS can be calculated once and used for all simulations.

        #N1_conc, N2_conc, M_conc = test_model.simple_ss_m(params_list= pars_list, M_ss = M_steady(), dt= dt,N = N)

        # N1_conc, N2_conc, M_conc = GRN.simple_ss(params_list= pars_list,dt=dt,N=N,M_ss=M_steady())
        # count = count+1
        # print('simulation complete', count)

        # N1_SS = N1_conc.loc[N1_conc.index[-1]] #Take steady state concentrations only
        # N2_SS = N2_conc.loc[N2_conc.index[-1]]

        M_SS = M_steady()
        N1_SS, N2_SS = GRN_steady(M_SS, par_list=pars_list)

        N1_thr = N1_SS.max()*0.95 #Obtain boundary poisition and corresponding morphogen weight.
        interp_func = interp1d(N1_SS,x)
        X1_thr = interp_func(N1_thr)
        X1_thr_list.append(X1_thr)

        M_interp_func = interp1d(x,M_SS)
        M_thr = M_interp_func(X1_thr)
        M1_thr_list.append(M_thr)

        N2_thr = N2_SS.max()*0.95 
        interp_func = interp1d(N2_SS,x)
        if N2_thr < N2_SS[0]:
            N2_thr = N2_SS[0]
        X2_thr = interp_func(N2_thr)
        X2_thr_list.append(X2_thr)

        M_thr = M_interp_func(X2_thr)
        M2_thr_list.append(M_thr)

    #Visualise the SS concentration profiles
    plt.plot(x,N1_SS, c = 'teal',label='node 1')
    plt.plot(x,N2_SS, c = 'grey', label= 'node 2')
    plt.plot(x,M_SS, c='darkorange', label = 'Morphogen')
    plt.title(f'{w_label} = {10**w}')
    plt.show()

    X1_thr_list = [a/L for a in X1_thr_list]
    X2_thr_list = [b/L for b in X2_thr_list]

    X_diff = [a-b for a,b in zip(X1_thr_list,X2_thr_list)]

    plt.plot(X1_thr_list, np.power(10, all_weights[0]), c = 'teal',label = 'N1')
    plt.yscale('log')
    plt.title(f'{species} weight dynamics')
    plt.plot(X2_thr_list, np.power(10, all_weights[0]), c = 'grey', label = 'N2')
    plt.xlabel('Xi threshold (x/L)')
    plt.ylabel(w_label)
    plt.show()

    plt.plot(M1_thr_list, np.power(10, all_weights[0]), c = 'teal')
    plt.title(f'{species} weight/morphogen dynamics')
    plt.plot(M2_thr_list, np.power(10, all_weights[0]), c = 'grey')
    plt.yscale('log')
    plt.xlabel('[Morphogen] at Xi threshold')
    plt.ylabel(w_label)
    plt.show()

    plt.plot(np.power(10, all_weights[0]),X_diff, c = 'teal',label = 'N1')
    plt.xscale('log')
    plt.title(f'{species} difference between X1 and X2')
    plt.xlabel(w_label)
    plt.ylabel('(X1/L)/(X2/L)')
    plt.show()


# %%
'''Analysing the relationship between two weights at the same time'''

def weights_pairwise(species:list, increment):
    S1= species[0]
    S2 = species[1]

    for weight in weights:
        if weight['Species'] == S1:
            S1_weights = np.arange(weight['lower_limit'],weight['upper_limit'],increment)
        if weight['Species'] == S2:
            S2_weights = np.arange(weight['lower_limit'],weight['upper_limit'],increment)

    #select the weights, do some starts with bs

    chr_s1 = S1[0] + S1[1]
    chr_s2 = S2[0] + S2[1]

    Data = pd.DataFrame({'S1_w':[],'S2_w':[],'X1_thr':[],'X2_thr':[]})
    

    par_dict = {'V_max_n1': 30, 
            'w_Mo': 1, 
            'w_N2':1, 
            'n_m': 1, 
            'n_n2': 1, 
            'V_max_n2': 30,
            'w_N1': 1, 
            'n_n1': 1, 
            'Deg_n1': 2, 
            'Deg_n2': 2}
    
    for s1w in S1_weights:
        par_dict['w_'+chr_s1] = 10**s1w
        for s2w in S2_weights:
            par_dict['w_'+chr_s2] = 10**s2w

            pars_list = list(par_dict.values())

            M_SS = M_steady()
            N1_SS, N2_SS = GRN_steady(M_SS, par_list=pars_list)

            N1_thr = N1_SS.max()*0.95 #Obtain boundary poisition and corresponding morphogen weight.
            interp_func = interp1d(N1_SS,x)
            X1_thr = interp_func(N1_thr)

            N2_thr = N2_SS.max()*0.95 
            interp_func = interp1d(N2_SS,x)
            if N2_thr < min(N2_SS):
                N2_thr = min(N2_SS)
            elif N2_thr > max(N2_SS):
                N2_thr = max(N2_SS)
            X2_thr = interp_func(N2_thr)

            temp = pd.DataFrame({'S1_w':[10**s1w],'S2_w':[10**s2w],'X1_thr':[X1_thr],'X2_thr':[X2_thr]})
            Data = pd.concat([Data,temp])

    #turn into rectangular df

    df = Data.groupby(by=['S1_w'])
    for j,(name,subdf) in enumerate(df):
        temp = pd.DataFrame(subdf['X1_thr'].to_list())
        temp = temp.T

        tempo = pd.DataFrame(subdf['X1_thr'])
        col_names = subdf['S2_w'].to_list()
        tempo.rename(columns={'X1_thr': name}, inplace= True)
        tempo = tempo.T
        tempo.columns = col_names


        temp2 = pd.DataFrame(subdf['X2_thr'].to_list())
        temp2 = temp2.T

        if j == 0:
            X1 = temp
            X1_summary = tempo
            X2 = temp2
        else:
            X1 = pd.concat([X1,temp],ignore_index=True)
            X2 = pd.concat([X2,temp2],ignore_index=True)
            X1_summary = pd.concat([X1_summary,tempo])
    X1 = X1.iloc[::-1] #reverses order
    X2 = X2.iloc[::-1]
    X1_summary = X1_summary.iloc[::-1]
    #plotting X1 heatmap
    c = plt.imshow(X1,extent=[-2,2,-2,2])
    c_bar = plt.colorbar(c)
    c_bar.set_label('Xi',rotation=270)
    plt.xlabel('w_'+chr_s2+' (log10)')
    plt.ylabel('w_'+chr_s1+' (log10)')
    plt.title(f'Steady state boundary position (X1) for {species[0]} & {species[1]} weights')
    plt.show()

    #plotting X2 heatmap
    c = plt.imshow(X2,extent=[-2,2,-2,2])
    c_bar = plt.colorbar(c)
    c_bar.set_label('Xi',rotation=270)
    plt.xlabel('w_'+chr_s2+' (log10)')
    plt.ylabel('w_'+chr_s1+' (log10)')
    plt.title(f'Steady state boundary position (X2) for {species[0]} & {species[1]} weights')
    plt.show()

        # temp = pd.DataFrame({'SW_1':[name],'X1_thr':[subdf['X1_thr'].to_list()]})
        # df1 = pd.concat([df1,temp])

    return X1, X2
#%%
'''Different size analysis'''
sizes = [2.8] #,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0]
y1 = 0.2
y2 = 0.8

weights_df = pd.DataFrame({'L':[], 'w_N1':[], 'w_N2':[],'y1':[], 'y2':[]})

for size in sizes:
    L = size
    K = 5.76 #linear degradation, for a decay length of 0.25 (4.8); 0.5 (1.2)
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
    tmax = 10/K
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
    
    X1, X2 = weights_pairwise(species= ['N1', 'N2'], increment = 0.02)
    X1 = X1
    X2 = X2
    x_lab = np.arange(-2,2,0.02)
    x_lab = [round(a,2) for a in x_lab]
    X1.columns = x_lab
    X2.columns = x_lab
    x_lab.reverse()
    x1 = X1.T
    x2 = X2.T
    x1.columns = x_lab
    x2.columns = x_lab
    X1 = x1.T
    X2 = x2.T

    #extracting weights for a specific X1 boundary
    w_n1, w_n2, x1, x2 = [], [], [], []

    for row in X1.index:
        for col in X1.columns:
            bound = X1[col][row]
            if bound >= (y1*size)*0.9999 and bound <= (y1*size) * 1.0001:
                w_n1.append(row)
                w_n2.append(col)
                x1.append(X1[col][row])
                x2.append(X2[col][row])
    df = pd.DataFrame({'W_n1': w_n1, 'W_n2': w_n2, 'X1': x1, 'X2':x2})
    size = str(size)
    size = size.replace('.', '_')
    df.to_csv(f'../results/2L{size}.csv')
    # #interpolate exact weights to give correct boundary positions and test it by plotting it.

    Interp = interp1d(df['X2'], df['W_n2'])
    target_x = y2*L 
    target_w_n2 = Interp(target_x)

    Interp = interp1d(df['X2'], df['W_n1'])
    target_w_n1 = Interp(target_x)

    par_dict = {'V_max_n1': 30, 
                'w_Mo': 1, 
                'w_N2':10**target_w_n2, 
                'n_m': 1, 
                'n_n2': 1, 
                'V_max_n2': 30,
                'w_N1': 10**target_w_n1, 
                'n_n1': 1, 
                'Deg_n1': 2, 
                'Deg_n2': 2}

    pars_list = list(par_dict.values())
    M_SS = M_steady()
    N1_SS, N2_SS = GRN_steady(M_SS, par_list=pars_list)

    N1_thr = max(N1_SS) * 0.95
    N2_thr = max(N2_SS) * 0.95

    interp_func = interp1d(N1_SS,x)
    X1_thr = interp_func(N1_thr)

    interp_func = interp1d(N2_SS,x)
    X2_thr = interp_func(N2_thr)


    plt.plot(x, M_SS, c = 'darkorange', label = 'morpho')
    plt.plot(x, N1_SS, c = 'teal', label = 'N1')
    plt.plot(x, N2_SS, c = 'grey', label = 'N2')
    plt.axvline(x = X1_thr, linestyle = 'dashed', c = 'teal', label = 'X1')
    plt.axvline(x = X2_thr, linestyle = 'dashed', c = 'grey', label = 'X2')
    plt.title(f'conc profiles @ SS, L = {L}')
    plt.xlabel('x')
    plt.ylabel('conc')
    plt.legend(bbox_to_anchor=(1.2, 1))
    plt.show()

    temp = pd.DataFrame({'L':[L], 'w_N1':[target_w_n1], 'w_N2':[target_w_n2],'y1':[X1_thr/L], 'y2':[X2_thr/L]})
    weights_df = pd.concat([weights_df,temp])
    print(f'L{L} complete')
#%%
#root finder to find position of x

#Function needs to take weights and produce a 

def min_func(log_weights:list,target_y:list,size):
    L = size
    K = 5.76 #linear degradation, for a decay length of 0.25 (4.8); 0.5 (1.2)
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
    tmax = 10/K
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

    par_dict = {'V_max_n1': 30, 
                'w_Mo': 1, 
                'w_N2':10**log_weights[1], 
                'n_m': 1, 
                'n_n2': 1, 
                'V_max_n2': 30,
                'w_N1': 10**log_weights[0], 
                'n_n1': 1, 
                'Deg_n1': 2, 
                'Deg_n2': 2}

    pars_list = list(par_dict.values())
    M_SS = M_steady()
    N1_SS, N2_SS = GRN_steady(M_SS, par_list=pars_list)

    N1_thr = max(N1_SS) * 0.95
    N2_thr = max(N2_SS) * 0.95

    interp_func = interp1d(N1_SS,x)
    est_X1 = interp_func(N1_thr)

    interp_func = interp1d(N2_SS,x)
    if N2_thr < min(N2_SS):
        N2_thr = min(N2_SS)
    elif N2_thr > max(N2_SS):
        N2_thr = max(N2_SS)
    est_X2 = interp_func(N2_thr)

    target_X1 = target_y[0]*L
    target_X2 = target_y[1]*L

    distance1 = np.power((target_X1 - est_X1),2) #Squared difference, smaller the difference, the closer it is to the minima.
    distance2 = np.power((target_X2 - est_X2),2)

    return distance1 + distance2 #problem with this is that one might be slightly better fit than the other. 

targ_y = [0.2,0.8]
siz = 2.8
result = minimize(min_func, args=(targ_y,siz), x0 = [0,0], bounds = [(-2,2),(-2,2)], options={"maxiter":1000,"disp":True}, method = 'Nelder-Mead')

weights = result.x










# %%
weights_df = pd.read_csv('../results/weights_df.csv')
plt.scatter(weights_df['w_N2'],weights_df['w_N1'])
plt.plot(weights_df['w_N2'],weights_df['w_N1'])
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.title('weight pairs for different L')
plt.xlabel('log(w_N2)')
plt.ylabel('log(w_N1)')
plt.show()

plt.scatter(weights_df['L'],weights_df['w_N2'], c ='grey')
plt.scatter(weights_df['L'],weights_df['w_N1'], c = 'teal')
plt.plot(weights_df['L'],weights_df['w_N2'], c ='grey', label = 'w_N2')
plt.plot(weights_df['L'],weights_df['w_N1'], c = 'teal', label = 'w_N1')
plt.xlim(2,4)
plt.ylim(-2,2)
plt.title('individual weight relationships with L')
plt.xlabel('L')
plt.ylabel('log(w_i)')
plt.legend(bbox_to_anchor = (1.2, 1))
plt.show()


# %%
L = 4.2
K = 5.76 #linear degradation, for a decay length of 0.25 (4.8); 0.5 (1.2)
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
tmax = 10/K
par_dict = {'V_max_n1': 30, 
            'w_Mo': 1, 
            'w_N2':10**-0.6771613499561298, 
            'n_m': 1, 
            'n_n2': 1, 
            'V_max_n2': 30,
            'w_N1': 10**0.9175298857644232, 
            'n_n1': 1, 
            'Deg_n1': 2, 
            'Deg_n2': 2}

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

pars_list = list(par_dict.values())
M_SS = M_steady()
N1_SS, N2_SS = GRN_steady(M_SS, par_list=pars_list)

N1_thr = max(N1_SS) * 0.95
N2_thr = max(N2_SS) * 0.95

interp_func = interp1d(N1_SS,x)
X1_thr = interp_func(N1_thr)

interp_func = interp1d(N2_SS,x)
X2_thr = interp_func(N2_thr)


plt.plot(x, M_SS, c = 'darkorange', label = 'morpho')
plt.plot(x, N1_SS, c = 'teal', label = 'N1')
plt.plot(x, N2_SS, c = 'grey', label = 'N2')
# plt.axvline(x = X1_thr, linestyle = 'dashed', c = 'teal', label = 'X1')
# plt.axvline(x = X2_thr, linestyle = 'dashed', c = 'grey', label = 'X2')
plt.title(f'conc profiles @ SS, L = {L}')
plt.xlabel('x')
plt.ylabel('conc')
plt.legend(bbox_to_anchor=(1.2, 1))
plt.show()

print(X1_thr, X2_thr, X1_thr/L, X2_thr/L)

# %%
