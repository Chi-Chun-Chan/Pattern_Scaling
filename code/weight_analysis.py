#%%
from GRNs import *
from Analytical_funcs import *
from typing import Any, Dict, List, Optional, Union
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
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

    X1_thr_list = [a for a in X1_thr_list]
    X2_thr_list = [b for b in X2_thr_list]

    X_diff = [a/b for a,b in zip(X1_thr_list,X2_thr_list)]

    plt.plot(X1_thr_list, all_weights[0], c = 'teal',label = 'X1')
    plt.title(f'{species} weight dynamics')
    plt.plot(X2_thr_list, all_weights[0], c = 'grey', label = 'X2')
    plt.xlabel('x')
    plt.ylabel(f'log({w_label})')
    plt.legend()
    plt.show()

    plt.plot(M1_thr_list, all_weights[0], c = 'teal')
    plt.title(f'{species} weight/morphogen dynamics')
    plt.plot(M2_thr_list, all_weights[0], c = 'grey')
    plt.xlabel('[Morphogen] at Xi threshold')
    plt.ylabel(f'log({w_label})')
    plt.legend()
    plt.show()

    plt.plot(X_diff, all_weights[0], c = 'black',label = 'X1/X2')
    plt.title(f'{species} difference between X1 and X2')
    plt.ylabel(f'log({w_label})')
    plt.xlabel('Xi ratio')
    plt.legend()
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

#renaming columns
#%%



# X1, X2 = weights_pairwise(species= ['N1', 'N2'], increment = 0.01)
# X1 = X1
# X2 = X2
# x_lab = np.arange(-2,2,0.01)
# x_lab = [round(a,2) for a in x_lab]
# X1.columns = x_lab
# X2.columns = x_lab
# x_lab.reverse()
# x1 = X1.T
# x2 = X2.T
# x1.columns = x_lab
# x2.columns = x_lab
# X1 = x1.T
# X2 = x2.T


# #extracting weights for a specific X1 boundary
# w_n1, w_n2, x1, x2 = [], [], [], []

# for row in X1.index:
#     for col in X1.columns:
#         bound = X1[col][row]
#         if bound >= 0.3999 and bound <= 0.4001:
#             w_n1.append(row)
#             w_n2.append(col)
#             x1.append(X1[col][row])
#             x2.append(X2[col][row])
# df2 = pd.DataFrame({'W_n1': w_n1, 'W_n2': w_n2, 'X1': x1, 'X2':x2})










































#%%
# L2 = pd.read_csv('../results/L2.csv')
# L14 = pd.read_csv('../results/L14.csv')


# #visualising weight combinations

# fig,ax = plt.subplots()
# ax.plot(L2['X1'], L2['W_n2'], c = 'teal', label='X1')
# ax.plot(L2['X2'], L2['W_n2'], c = 'grey', label = 'X2')
# ax.set_xlim(0,2.0)
# ax.set_ylim(-2,2)
# ax.set_ylabel('w_n2(log)')
# ax.set_xlabel('x')
# ax.set_title('Boundary position for X1= c isoline')
# ax.legend()
# plt.show()

# fig,ax = plt.subplots()
# ax.plot(L2['X1'], L2['W_n1'], c = 'teal', label='X1')
# ax.plot(L2['X2'], L2['W_n1'], c = 'grey', label = 'X2')
# ax.set_xlim(0,2.0) #change this
# ax.set_ylim(-2,2)
# ax.set_ylabel('w_n2(log)')
# ax.set_xlabel('x')
# ax.set_title('Boundary position for X1=c isoline')
# ax.legend()
# plt.show()

# #interpolate exact weights to give correct boundary positions and test it by plotting it.

# Interp = interp1d(L2['X2'], L2['W_n2'])
# target_x = 1.6 #L2= 1.6, L24= 2.08, L14 = 1.19
# target_w_n2 = Interp(target_x)

# Interp = interp1d(L2['X2'], L2['W_n1'])
# target_w_n1 = Interp(target_x)

# par_dict = {'V_max_n1': 30, 
#             'w_Mo': 1, 
#             'w_N2':10**target_w_n2, 
#             'n_m': 1, 
#             'n_n2': 1, 
#             'V_max_n2': 30,
#             'w_N1': 10**target_w_n1, 
#             'n_n1': 1, 
#             'Deg_n1': 2, 
#             'Deg_n2': 2}

# pars_list = list(par_dict.values())
# M_SS = M_steady()
# N1_SS, N2_SS = GRN_steady(M_SS, par_list=pars_list)

# N1_thr = max(N1_SS) * 0.95
# N2_thr = max(N2_SS) * 0.95

# interp_func = interp1d(N1_SS,x)
# X1_thr = interp_func(N1_thr)

# interp_func = interp1d(N2_SS,x)
# X2_thr = interp_func(N2_thr)


# plt.plot(x, M_SS, c = 'darkorange', label = 'morpho')
# plt.plot(x, N1_SS, c = 'teal', label = 'N1')
# plt.plot(x, N2_SS, c = 'grey', label = 'N2')
# plt.axvline(x = X1_thr, linestyle = 'dashed', c = 'teal', label = 'X1')
# plt.axvline(x = X2_thr, linestyle = 'dashed', c = 'grey', label = 'X2')
# plt.title('conc profiles @ SS, L = 2.0')
# plt.xlabel('x')
# plt.ylabel('conc')
# plt.legend(bbox_to_anchor=(1.2, 1))
# plt.show()

# print(f'X1 = {X1_thr}, X2 = {X2_thr}, w_n1(log) = {target_w_n1}, w_n2(log) = {target_w_n2}')

# plt.plot(x/L, M_SS, c = 'darkorange', label = 'morpho')
# plt.plot(x/L, N1_SS, c = 'teal', label = 'N1')
# plt.plot(x/L, N2_SS, c = 'grey', label = 'N2')
# plt.axvline(x = X1_thr/L, linestyle = 'dashed', c = 'teal', label = 'X1')
# plt.axvline(x = X2_thr/L, linestyle = 'dashed', c = 'grey', label = 'X2')
# plt.title('conc profiles @ SS, L = 2.0')
# plt.xlabel('y = x/L')
# plt.ylabel('conc')
# plt.legend(bbox_to_anchor=(1.2, 1))
# plt.show()

# #weight x weight relationship
# # plt.plot(L26['W_n2'], L26['W_n1'], label = 'L = 2.6')
# # plt.xlim(-2,2)
# # plt.ylim(-2,2)
# # plt.xlabel('w_N2')
# # plt.ylabel('w_N1')
# # plt.plot(L14['W_n2'], L14['W_n1'], label = 'L = 1.4')
# # plt.plot(L2['W_n2'], L2['W_n1'], label = 'L = 2.0')
# # plt.legend(bbox_to_anchor=(1.2,1))


# #%%
# #%%
# L14 = pd.read_csv('../results/L14.csv')


# #visualising weight combinations

# fig,ax = plt.subplots()
# ax.plot(L14['X1'], L14['W_n2'], c = 'teal', label='X1')
# ax.plot(L14['X2'], L14['W_n2'], c = 'grey', label = 'X2')
# ax.set_xlim(0,1.4)
# ax.set_ylim(-2,2)
# ax.set_ylabel('w_n2(log)')
# ax.set_xlabel('x')
# ax.set_title('Boundary position for X1= c isoline')
# ax.legend()
# plt.show()

# fig,ax = plt.subplots()
# ax.plot(L14['X1'], L14['W_n1'], c = 'teal', label='X1')
# ax.plot(L14['X2'], L14['W_n1'], c = 'grey', label = 'X2')
# ax.set_xlim(0,1.4) #change this
# ax.set_ylim(-2,2)
# ax.set_ylabel('w_n2(log)')
# ax.set_xlabel('x')
# ax.set_title('Boundary position for X1=c isoline')
# ax.legend()
# plt.show()

# #interpolate exact weights to give correct boundary positions and test it by plotting it.

# Interp = interp1d(L14['X2'], L14['W_n2'])
# target_x = 1.119 #L2= 1.6, L24= 2.08, L14 = 1.119
# target_w_n2 = Interp(target_x)

# Interp = interp1d(L14['X2'], L14['W_n1'])
# target_w_n1 = Interp(target_x)

# par_dict = {'V_max_n1': 30, 
#             'w_Mo': 1, 
#             'w_N2':10**target_w_n2, 
#             'n_m': 1, 
#             'n_n2': 1, 
#             'V_max_n2': 30,
#             'w_N1': 10**target_w_n1, 
#             'n_n1': 1, 
#             'Deg_n1': 2, 
#             'Deg_n2': 2}

# pars_list = list(par_dict.values())
# M_SS = M_steady()
# N1_SS, N2_SS = GRN_steady(M_SS, par_list=pars_list)

# N1_thr = max(N1_SS) * 0.95
# N2_thr = max(N2_SS) * 0.95

# interp_func = interp1d(N1_SS,x)
# X1_thr = interp_func(N1_thr)

# interp_func = interp1d(N2_SS,x)
# X2_thr = interp_func(N2_thr)


# plt.plot(x, M_SS, c = 'darkorange', label = 'morpho')
# plt.plot(x, N1_SS, c = 'teal', label = 'N1')
# plt.plot(x, N2_SS, c = 'grey', label = 'N2')
# plt.axvline(x = X1_thr, linestyle = 'dashed', c = 'teal', label = 'X1')
# plt.axvline(x = X2_thr, linestyle = 'dashed', c = 'grey', label = 'X2')
# plt.title('conc profiles @ SS, L = 1.4')
# plt.xlabel('x')
# plt.ylabel('conc')
# plt.legend(bbox_to_anchor=(1.2, 1))
# plt.show()

# print(f'X1 = {X1_thr}, X2 = {X2_thr}, w_n1(log) = {target_w_n1}, w_n2(log) = {target_w_n2}')

# plt.plot(x/L, M_SS, c = 'darkorange', label = 'morpho')
# plt.plot(x/L, N1_SS, c = 'teal', label = 'N1')
# plt.plot(x/L, N2_SS, c = 'grey', label = 'N2')
# plt.axvline(x = X1_thr/L, linestyle = 'dashed', c = 'teal', label = 'X1')
# plt.axvline(x = X2_thr/L, linestyle = 'dashed', c = 'grey', label = 'X2')
# plt.title('conc profiles @ SS, L = 1.4')
# plt.xlabel('y = x/L')
# plt.ylabel('conc')
# plt.legend(bbox_to_anchor=(1.2, 1))
# plt.show()

# #weight x weight relationship
# # plt.plot(L26['W_n2'], L26['W_n1'], label = 'L = 2.6')
# # plt.xlim(-2,2)
# # plt.ylim(-2,2)
# # plt.xlabel('w_N2')
# # plt.ylabel('w_N1')
# # plt.plot(L14['W_n2'], L14['W_n1'], label = 'L = 1.4')
# # plt.plot(L2['W_n2'], L2['W_n1'], label = 'L = 2.0')
# # plt.legend(bbox_to_anchor=(1.2,1))


# #%%
# L26 = pd.read_csv('../results/L26.csv')

# #visualising weight combinations

# fig,ax = plt.subplots()
# ax.plot(L26['X1'], L26['W_n2'], c = 'teal', label='X1')
# ax.plot(L26['X2'], L26['W_n2'], c = 'grey', label = 'X2')
# ax.set_xlim(0,2.6)
# ax.set_ylim(-2,2)
# ax.set_ylabel('w_n2(log)')
# ax.set_xlabel('x')
# ax.set_title('Boundary position for X1= c isoline')
# ax.legend()
# plt.show()

# fig,ax = plt.subplots()
# ax.plot(L26['X1'], L26['W_n1'], c = 'teal', label='X1')
# ax.plot(L26['X2'], L26['W_n1'], c = 'grey', label = 'X2')
# ax.set_xlim(0,2.6) #change this
# ax.set_ylim(-2,2)
# ax.set_ylabel('w_n2(log)')
# ax.set_xlabel('x')
# ax.set_title('Boundary position for X1=c isoline')
# ax.legend()
# plt.show()

# #interpolate exact weights to give correct boundary positions and test it by plotting it.

# Interp = interp1d(L26['X2'], L26['W_n2'])
# target_x = 2.08 #L2= 1.6, L24= 2.08, L14 = 1.19
# target_w_n2 = Interp(target_x)

# Interp = interp1d(L26['X2'], L26['W_n1'])
# target_w_n1 = Interp(target_x)

# par_dict = {'V_max_n1': 30, 
#             'w_Mo': 1, 
#             'w_N2':10**target_w_n2, 
#             'n_m': 1, 
#             'n_n2': 1, 
#             'V_max_n2': 30,
#             'w_N1': 10**target_w_n1, 
#             'n_n1': 1, 
#             'Deg_n1': 2, 
#             'Deg_n2': 2}

# pars_list = list(par_dict.values())
# M_SS = M_steady()
# N1_SS, N2_SS = GRN_steady(M_SS, par_list=pars_list)

# N1_thr = max(N1_SS) * 0.95
# N2_thr = max(N2_SS) * 0.95

# interp_func = interp1d(N1_SS,x)
# X1_thr = interp_func(N1_thr)

# interp_func = interp1d(N2_SS,x)
# X2_thr = interp_func(N2_thr)


# plt.plot(x, M_SS, c = 'darkorange', label = 'morpho')
# plt.plot(x, N1_SS, c = 'teal', label = 'N1')
# plt.plot(x, N2_SS, c = 'grey', label = 'N2')
# plt.axvline(x = X1_thr, linestyle = 'dashed', c = 'teal', label = 'X1')
# plt.axvline(x = X2_thr, linestyle = 'dashed', c = 'grey', label = 'X2')
# plt.title('conc profiles @ SS, L = 2.6')
# plt.xlabel('x')
# plt.ylabel('conc')
# plt.legend(bbox_to_anchor=(1.2, 1))
# plt.show()

# print(f'X1 = {X1_thr}, X2 = {X2_thr}, w_n1(log) = {target_w_n1}, w_n2(log) = {target_w_n2}')

# #%%
# # X2.replace(to_replace = 0.0110922271, value = 0 , inplace= True)
# # X2.replace(to_replace = 0.0709224837, value = 0 , inplace= True)
# relative_difference = (X1) - (X2)
# x_lab = np.arange(-2,2,0.04)
# x_lab = [round(a,2) for a in x_lab]
# relative_difference.columns = x_lab
# x_lab.reverse()
# rd = relative_difference.T
# rd.columns = x_lab
# relative_difference = rd.T
# # relative_difference.replace(to_replace = 13.9215423825, value = 0/2 , inplace= True)
# c = plt.imshow(relative_difference,extent=[-2,2,-2,2])
# c_bar = plt.colorbar(c)
# c_bar.set_label('(X1/L) - (X2/L) ',rotation=270)
# plt.xlabel('w_n2')
# plt.ylabel('w_n1')
# plt.title(f'Relative difference between X1 and X2 boundaries')
# plt.show()

# relative_difference.to_csv('../results/relative_frac_N1N2.csv')
            

# x_lab = np.arange(-2,2,0.05)
# x_lab = [round(a,2) for a in x_lab]
# # y_lab = np.arange(-2,2,0.1)
# # y_lab = [round(a,2) for a in y_lab]
# # y_lab.reverse()
# # sns.set(rc={'figure.figsize':(25.7,14.27)})
# # ax = sns.heatmap(relative_difference, annot=True,fmt=".3f", xticklabels = x_lab, yticklabels = y_lab)
# # ax.set(xlabel="w_n2", ylabel="w_n1", title='relative fraction (X1/X2)')
# # ax.xaxis.tick_top()

# c1 = X1[20].to_list() #change these values
# c2 = X1[40].to_list()
# c3 = X1[60].to_list()
# c12 = X2[20].to_list()
# c22 = X2[40].to_list()
# c32 = X2[60].to_list()
# c1.reverse()
# c2.reverse()
# c3.reverse()
# c12.reverse()
# c22.reverse()
# c32.reverse()
# plt.plot(c1,x_lab, c = 'teal', label = 'w_n2 = 0.1', alpha = 0.3)
# plt.plot(c2,x_lab, c = 'teal', label = 'w_n2 = 1', alpha = 0.7)
# plt.plot(c3, x_lab, c = 'teal', label = 'w_n2 = 10', alpha = 1)
# plt.plot(c12,x_lab, c = 'grey', label = 'w_n2 = 0.1', alpha = 0.3)
# plt.plot(c22,x_lab, c = 'grey', label = 'w_n2 = 1', alpha = 0.7)
# plt.plot(c32,x_lab, c = 'grey', label = 'w_n2 = 10', alpha = 1)
# plt.title('Exploring relative boundary patterning for different weight constants')
# plt.ylabel('weights (w_N1)')
# plt.xlabel('Xi')
# plt.legend()
# plt.show()

# c_1 = (X1[20] - X2[20]).to_list()
# c_2 = (X1[40] - X2[40]).to_list()
# c_3 = (X1[60] - X2[60]).to_list()
# c_1.reverse()
# c_2.reverse()
# c_3.reverse()


# plt.plot(c_1,x_lab, label = 'w_n2 = 0.1', linestyle = 'dashed')
# plt.plot(c_2,x_lab, label = 'w_n2 = 1', linestyle = 'dashed')
# plt.plot(c_3,x_lab, label = 'w_n2 = 10', linestyle = 'dashed')
# plt.title('Relative difference between boundaries for different weight constants')
# plt.xlabel('X1-X2')
# plt.ylabel('weights (w_N1)')
# plt.legend()


# # %%

# #relative_difference
# #labelling of dataframes
# x_lab = np.arange(-2,2,0.05)
# x_lab = [round(a,2) for a in x_lab]
# X1.columns = x_lab
# X2.columns = x_lab
# x_lab.reverse()
# x1 = X1.T
# x2 = X2.T
# x1.columns = x_lab
# x2.columns = x_lab
# X1 = x1.T
# X2 = x2.T

# #Iterate over heatmap and isolate pairs of weights which produce the correct distance.
# w_n1, w_n2, x1mx2, x1, x2 = [], [], [], [], []
# for row in relative_difference.index:
#     for col in relative_difference.columns:
#         diff = relative_difference[col][row]
#         if diff <= -0.595 and diff >= -0.605: #lets set constant difference as -0.6, -0.595,-0.605
#             x1mx2.append(diff)
#             w_n1.append(row)
#             w_n2.append(col)
#             x1.append(X1[col][row])
#             x2.append(X2[col][row])


# df = pd.DataFrame({'W_n1': w_n1, 'W_n2': w_n2, 'X1-X2': x1mx2, 'X1': x1, 'X2':x2})

# plt.scatter(df['W_n2'], df['W_n1'])
# plt.ylabel('w_n1')
# plt.xlabel("w_n2")
# plt.title('relationship between weights for constant X1-X2 = -0.6')
# plt.show()

# plt.scatter(df['X1-X2'],df['W_n2'], label = 'Xdiff')
# plt.scatter(df['X1'],df['W_n2'], label = 'X1')
# plt.scatter(df['X2'],df['W_n2'], label = 'X2')
# plt.title('Boundaries and their differences')
# plt.legend()
# plt.ylabel('w_n2')
# plt.show()
# plt.scatter(df['X1-X2'],df['W_n1'], label = 'Xdiff')
# plt.scatter(df['X1'],df['W_n1'], label = 'X1')
# plt.scatter(df['X2'],df['W_n1'], label = 'X2')
# plt.title('Boundaries and their differences')
# plt.ylabel('w_n1')
# plt.legend()


# #for loop which loops through all values and stores weights and X1-X2 = 0.15 +- 0.01.
# # %%
# df2 = pd.read_csv('../data/wm2_xneg06_df_L2.csv')
# df1 = pd.read_csv('../data/wm1_xneg06_df_L2.csv')
# df_point5 = pd.read_csv('../data/wm05_xneg06_df_L2.csv')

# plt.scatter(df_point5['W_n2'], df_point5['W_n1'], label = 'w_m = 0.5')
# plt.scatter(df1['W_n2'], df1['W_n1'],  label = 'w_m = 1')
# plt.scatter(df2['W_n2'], df2['W_n1'],  label = 'w_m = 2')
# plt.ylabel('w_n1')
# plt.xlabel("w_n2")
# plt.title('relationship between weights for different w_m (constant X1-X2 = -0.6)')
# plt.legend()
# plt.show()

# plt.scatter(df_point5['X1'], df_point5['W_n1'],  label = 'w_m = 0.5')
# plt.scatter(df1['X1'], df1['W_n1'],  label = 'w_m = 1')
# plt.scatter(df2['X1'], df2['W_n1'],  label = 'w_m = 2')
# plt.ylabel('w_n1')
# plt.xlabel('x/L')
# plt.title('Boundary position X1 for different w_m')
# plt.legend()
# plt.show()

# plt.scatter(df_point5['X2'], df_point5['W_n1'],  label = 'w_m = 0.5')
# plt.scatter(df1['X2'], df1['W_n1'],  label = 'w_m = 1')
# plt.scatter(df2['X2'], df2['W_n1'],  label = 'w_m = 2')
# plt.ylabel('w_n1')
# plt.xlabel('x/L')
# plt.title('Boundary position X2 for different w_m')
# plt.legend()
# plt.show()

# plt.scatter(df_point5['X1'], df_point5['W_n1'],  label = 'w_m = 0.5')
# plt.scatter(df1['X1'], df1['W_n2'],  label = 'w_m = 1')
# plt.scatter(df2['X1'], df2['W_n2'],  label = 'w_m = 2')
# plt.ylabel('w_n2')
# plt.xlabel('x/L')
# plt.title('Boundary position X2 for different w_m')
# plt.legend()
# plt.show()

# plt.scatter(df_point5['X2'], df_point5['W_n2'],  label = 'w_m = 0.5')
# plt.scatter(df1['X2'], df1['W_n2'],  label = 'w_m = 1')
# plt.scatter(df2['X2'], df2['W_n2'],  label = 'w_m = 2')
# plt.ylabel('w_n1')
# plt.xlabel('x/L')
# plt.title('Boundary position X2 for different w_m')
# plt.legend()
# plt.show()




# %%
