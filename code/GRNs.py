#%%
import numpy as np
import pandas as pd
from Global_terms import *
from Equations import *
import progressbar as pb

# Steady state for morphogen
                    # for v1,v2 in zip(M_log.loc[count-1], M_log.loc[count]):
                    #     difference = v2-v1
                    #     if abs(difference) > max:
                    #         max = abs(difference)
                    #         ratio = max/v1
                    # if ratio < threshold:
                    #     Steady = True

                    #Steady state of GRN rather than morphogen
                    # for v1,v2 in zip(N1_log.loc[N1_log.index[count-1]], N1_log.loc[N1_log.index[count]]): 
                    #     difference = v2-v1
                    #     if abs(difference) > max:
                    #         max = abs(difference)
                    #         if v1 == 0:
                    #             v1 = 1
                    #         ratio = max/(v1)
                    # if ratio < threshold:
                    #     Steady_n1 = True
                    # for v1,v2 in zip(N2_log.loc[N2_log.index[count-1]], N2_log.loc[N2_log.index[count]]):
                    #     difference = v2-v1
                    #     if abs(difference) > max: 
                    #         max = abs(difference)
                    #         ratio = max/v1
                    # if ratio < threshold:
                    #     Steady_n2 = True
                    # if Steady_n1 == True & Steady_n2 == True:
                    #     Steady = True
                    #Assumed steady state limit.
#%%
class test_model:
    def __init__(self,params_list:list,M_conc):
        self.params_list=params_list
        self.M_conc=M_conc

    @staticmethod
    def simple(params_list,M_conc): #Assumes steady state at the given M_conc and solves for N1 and N2 analytically
        V_max_n1 = params_list[0]
        w_m = params_list[1] 
        w_n2 = params_list[2]
        n_m = params_list[3]
        n_n2 = params_list[4]

        V_max_n2 = params_list[5]
        w_n1 = params_list[6]
        n_n1 = params_list[7]

        Deg_n1 = params_list[8]  
        Deg_n2 = params_list[9]

        #Assuming steady state, solve for N1 analytically.

        N1 = V_max_n1 * np.power((w_m*M_conc),n_m)
        N1 /= 1 + np.power((w_m*M_conc),n_m) + np.power((w_n2*N2),n_n2) 
        N1 /= Deg_n1

        N1 = V_max_n1 * np.power((w_m*M_conc),n_m)
        N1 /= 1 + np.power((w_m*M_conc),n_m) + np.power((w_n2*(1/w_n2*(V_max_n1*np.power((w_m*M_conc),n_m))/Deg_n1*N1 -1 - np.power((w_m*M_conc),n_m))^(1/n_n1)),n_n2) 
        N1 /= Deg_n1

        N2 = V_max_n2
        N2 /= 1 + np.power((w_n1*N1),n_n1)
        N2 /= Deg_n2

        return N1, N2, M_conc #Assumes steady state at each morphogen gradient
    
    @staticmethod
    def simple_dynamical(params_list,dt,N):
        '''ODEs for each node which grows with the concentration profile'''
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

        #Initial conditions
        M0 = np.zeros(len(x))
        N1_0 = np.zeros(len(x))
        N2_0 = np.zeros(len(x))

        d = D/(dx**2)
        length = len(x)
        Sp = int(source_bound + 1)
        Nm = N - 1
        dCdt = np.zeros(int(length))  

        def PDE_M(c): 
            """ PDE solver, requires C(x) for a given t """
            dCdt[0] = d * (2*(c[1] - c[0])) - K*c[0] + v
            for i in range(1, Sp):
                dCdt[i] = d * (c[i+1] - 2*c[i] + c[i-1]) - K*c[i] + v
            for i in range(Sp, N):
                dCdt[i] = d * (c[i+1] - 2*c[i] + c[i-1]) - K*c[i]
            dCdt[N] = d * (2*(c[Nm] - c[N])) - K*c[N]
            # dCdt[int(length)-1] = d * (2*(c[int(length)-2] - c[int(length)-1])) - K*c[int(length)-1]

            return c + dCdt*dt

        def ODE_N1(n1,conc,N2t):
            dN1dt = V_max_n1 * np.power((w_m*conc),n_m)
            dN1dt /= (1 + np.power((w_m*conc),n_m) + np.power((w_n2*N2t),n_n2)) 
            dN1dt -= (Deg_n1 * n1) 
            return n1 + dN1dt * dt #returns N1 concentration after every dt
        
        def ODE_N2(n2,N1t):
            dN2dt = V_max_n2
            dN2dt /= 1 + np.power((w_n1*N1t),n_n1)
            dN2dt -= (Deg_n2 * n2)
            return n2 + dN2dt*dt
        
        t_passed = 0
        t_passed_all = 0
        threshold = 10e-6 #minimum amount of change between timesteps to be considered as steady state.
        Steady = False
        count = 0
        SS_time = 10/K

        Mt = PDE_M(M0)
        N1_t = ODE_N1(N1_0,M0,N2_0)
        N2_t = ODE_N2(N2_0, N1_0)

        N1_tc = ODE_N1(N1_t,Mt,N2_t) #use previous timestep
        N2_t = ODE_N2(N2_t,N1_t) #use previous timestep for N1_t
        Mt = PDE_M(Mt)
        N1_t = N1_tc #starting at non-zero values mitigates the need to check.

        M_log = pd.DataFrame(Mt) #Storage of concentration profiles
        M_log = M_log.transpose()
        N1_log = pd.DataFrame(N1_t) 
        N1_log = N1_log.transpose()
        N2_log = pd.DataFrame(N2_t) 
        N2_log = N2_log.transpose()

        #bar = pb.ProgressBar(maxval = (100/Deg_n1)).start()

        while t_passed_all <= SS_time:
            # Mt = PDE_M(Mt)
            N1_tc = ODE_N1(N1_t,Mt,N2_t) #use previous timestep
            N2_t = ODE_N2(N2_t,N1_t) #use previous timestep for N1_t
            Mt = PDE_M(Mt)
            N1_t = N1_tc #Now update previous N1 with new value

            t_passed += dt
            t_passed_all += t_passed
            count += 1
            if count % 10 == 0:
                M_log.loc[len(M_log)] = Mt #stores list of concentration into array
                N1_log.loc[len(N1_log)] = N1_t
                N2_log.loc[len(N2_log)] = N2_t
                t_passed = 0
            # bar.update(t_passed_all)
        
        # while (Steady_n1 == 1 & Steady_n2 == 1) == False:
        while Steady == False:
            N1_tc = ODE_N1(N1_t,Mt,N2_t) #use previous timestep
            N2_t = ODE_N2(N2_t,N1_t) #use previous timestep for N1_t
            Mt = PDE_M(Mt)
            N1_t = N1_tc
            t_passed += dt

            if t_passed >= tmax/(N_t): #Checks every time step
                count += 1
                if count % 10 == 0:
                    M_log.loc[len(M_log)] = Mt #stores list of concentration into array
                    N1_log.loc[len(N1_log)] = N1_t
                    N2_log.loc[len(N2_log)] = N2_t

                    N1_diff = [abs((a - b)/a) for a, b in zip(N1_log.loc[N1_log.index[-2]], N1_log.loc[N1_log.index[-1]])]
                    N2_diff = [abs((a - b)/a) for a, b in zip(N1_log.loc[N1_log.index[-2]], N1_log.loc[N1_log.index[-1]])]

                    if max(N1_diff) <= threshold and max(N2_diff) <= threshold:
                        Steady = True
                t_passed_all += t_passed
                # bar.update(t_passed_all)
                #if t_passed_all > SS_time:
                    #Steady = True
                t_passed = 0
            
        return N1_log, N2_log, M_log
    @staticmethod
    def simple_ss(params_list,dt,N):
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

        d = D/(dx**2)
        length = len(x)
        Lm = int(length - 1)
        Sp = int(source_bound + 1)
        Nm = N - 1
        dCdt = np.zeros(int(length))      

        def PDE_M(c): 
            """ PDE solver, requires C(x) for a given t """
            dCdt[0] = d * (2*(c[1] - c[0])) - K*c[0] + v
            for i in range(1,Sp):
                dCdt[i] = d * (c[i+1] - 2*c[i] + c[i-1]) - K*c[i] + v
            for i in range(Sp, N):
                dCdt[i] = d * (c[i+1] - 2*c[i] + c[i-1]) - K*c[i]
            dCdt[N] = d * (2*(c[Nm] - c[N])) - K*c[N]
            # dCdt[int(length)-1] = d * (2*(c[int(length)-2] - c[int(length)-1])) - K*c[int(length)-1 
            return c + dCdt*dt

        def ODE_N1(n1,conc,N2t):
            dN1dt = V_max_n1 * np.power((w_m*conc),n_m)
            dN1dt /= (1 + np.power((w_m*conc),n_m) + np.power((w_n2*N2t),n_n2)) 
            dN1dt -= Deg_n1 * n1 
            return n1 + dN1dt * dt #returns N1 concentration after every dt
        
        def ODE_N2(n2,N1t):
            dN2dt = V_max_n2
            dN2dt /= 1 + np.power((w_n1*N1t),n_n1)
            dN2dt -= Deg_n2 * n2
            return n2 + dN2dt*dt
    
        
        t_passed = 0
        t_passed_all = 0
        threshold = 0.001 #minimum amount of change between timesteps to be considered as steady state.
        Steady = False
        Steady_n1 = 0
        Steady_n2 = 0
        count = 1
        SS_time = 10/Deg_n1

        #Initial conditions
        M0 = np.zeros(len(x))
        N1_0 = np.zeros(len(x))
        N2_0 = np.zeros(len(x))

        Mt = PDE_M(M0)
        N1_t = ODE_N1(N1_0,M0,N2_0)
        N2_t = ODE_N2(N2_0, N1_0)

        Mt = PDE_M(Mt)
        N1_t = ODE_N1(N1_t,Mt,N2_t)
        N2_t = ODE_N2(N2_t,N1_t) #starting at non-zero values mitigates the need to check.
    
        bar = pb.ProgressBar(maxval =SS_time+100*dt).start()
        
        # while (Steady_n1 == 1 & Steady_n2 == 1) == False:
        while t_passed <= SS_time:
            Mt = PDE_M(Mt)
            N1_t = ODE_N1(N1_t,Mt,N2_t)
            N2_t = ODE_N2(N2_t,N1_t)
            t_passed += dt
            bar.update(t_passed)

        return Mt, N1_t, N2_t
    @staticmethod
    def simple_ss_m(params_list,M_ss,dt,N):
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

        #arbitrary time point to integrate ODE up to
        t = np.linspace(0,2,2) 
        t_passed = 0
        t_passed_all = 0
        threshold = 0.001
        Steady = False
        Steady_n1 = False
        Steady_n2 = False
        count = 0
        SS_time = 20/Deg_n1

        # d = D/(dx**2)
        # length = L/dx
        # Lm = int(length - 1)
        # Sp = int(source_bound + 1)
        # Nm = N - 1
        # dCdt = np.zeros(int(length))      

        # def PDE_M(c, t): 
        #     """ PDE solver, requires C(x) for a given t """
        #     dCdt[0] = d * (2*(c[1] - c[0])) - K*c[0] + v
        #     for i in range(1,int(source_bound+1)):
        #         dCdt[i] = d * (c[i+1] - 2*c[i] + c[i-1]) - K*c[i] + v
        #     for i in range(int(source_bound+1), N):
        #         dCdt[i] = d * (c[i+1] - 2*c[i] + c[i-1]) - K*c[i]
        #     dCdt[N] = d * (2*(c[Nm] - c[N])) - K*c[N]
        #     # dCdt[int(length)-1] = d * (2*(c[int(length)-2] - c[int(length)-1])) - K*c[int(length)-1]

        #     return c + dCdt*dt

        def ODE_N1(n1,conc,N2t):
            dN1dt = V_max_n1 * np.power((w_m*conc),n_m)
            dN1dt /= (1 + np.power((w_m*conc),n_m) + np.power((w_n2*N2t),n_n2)) 
            dN1dt -= Deg_n1 * n1 
            return n1 + dN1dt * dt #returns N1 concentration after every dt
        
        def ODE_N2(n2,N1t):
            dN2dt = V_max_n2
            dN2dt /= 1 + np.power((w_n1*N1t),n_n1)
            dN2dt -= Deg_n2 * n2
            return n2 + dN2dt*dt
        
        #initial conditions
        N1_0 = np.zeros(N+1)
        N2_0 = np.zeros(N+1)


        N1_t = ODE_N1(N1_0,M_ss,N2_0) #morphogen concentrations at steady state as initial point.
        N2_t = ODE_N2(N2_0,N1_0)
        M_t = M_ss

        M_log = pd.DataFrame(M_ss) 
        M_log = M_log.transpose()
        N1_log = pd.DataFrame(N1_t) 
        N1_log = N1_log.transpose()
        N2_log = pd.DataFrame(N2_t) 
        N2_log = N2_log.transpose()

        bar = pb.ProgressBar(maxval = (SS_time + 1.2*tmax/(N_t))).start()


        while Steady == False:
            N1_t = ODE_N1(N1_t,M_ss,N2_t)
            N2_t = ODE_N2(N2_t,N1_t)
            t_passed += dt

            if t_passed >= tmax/(N_t): #Checks every time step
                count += 1 #stores list of concentration into array                
                if count % 10 == 0:
                    N1_log.loc[len(N1_log)] = N1_t
                    N2_log.loc[len(N2_log)] = N2_t
                    M_log.loc[len(M_log)] = M_ss
                
                #Assumed steady state limit.
                t_passed_all += t_passed
                bar.update(t_passed_all)
                if t_passed_all > SS_time:
                    Steady = True
                t_passed = 0

        return N1_log, N2_log, M_log


    

# %%