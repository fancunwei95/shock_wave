import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.optimize import newton
from problem_2 import get_array_from_eta

## parameters ###################
gamma = 5.0/3.0
U_p = 20.0
P_p = 1.0   #20.0
vis_a = 1.8 
vis_b = 0.2
alpha = 3.0
n_p = 2000   #20000
n_t = 100000
n_c  = 2000
ratio_of_critical = 0.15
plot = True
## functions #####################
# u, R, V, e, P, C, q,  t, V_0, r, alpha, vis_a, vis_b, gamma

def get_initial_r(V, alpha, n_p, parameter_1D = 10.0, parameter_2D = 1.0):
    r = np.zeros(n_p)
    factor = [ parameter_1D, parameter_2D*np.pi, np.pi]
    def f(dr,r_1, alpha ,target, factor):
        return dr*(2*r_1+dr)**(alpha-1)* factor[int(alpha-1)] - target
    if alpha == 1:
        r[1] = V[0]/factor[0] 
    if alpha == 2:
        r[1] = np.sqrt(V[0]/(factor[1]))
    if alpha == 3:
        r[1] = (V[0]/factor[2]*3.0/4.0)**(1.0/3.0)
    dr_0 = (V[0]/factor[int(alpha)-1])**(1.0/alpha)
    
    for i in range(1, len(r)-1):
        r_1 = r[i]
        target = V[i]
        dr = newton( f, dr_0,  fprime = None, args=(r_1,alpha, target, factor), tol = 1.0e-16 )
        r[i+1] = r[i] + dr
    #print V[:-1]/((r[:-1]+r[1:])**2)/(r[1:]-r[:-1])/np.pi
    return r

def initial(U_p, P_p, n_p, n_t, gamma, alpha ):
    u = np.zeros((2, n_p))
    R = np.zeros((2, n_p))
    V = np.zeros((2, n_p))
    e = np.zeros((2, n_p))
    P = np.zeros((2, n_p))
    C = np.zeros((2, n_p))
    t = np.zeros(2)
    
    V_0  = np.zeros(n_p)+0.1                        # initial V
    r= get_initial_r(V_0,alpha,n_p)                 # initial r
    V[0] = np.zeros(n_p)+0.1                        # initialize V
    R[0] = get_initial_r(V_0,alpha,n_p)             # initialize R
    u[0][-1] = U_p                 		    # initialize u
    P[0][:n_c] = np.zeros(n_c)+P_p                  # initialize P
    C[0] = np.sqrt(gamma*P[0]*V[0])
    e[0] = 1.0/(gamma - 1.0)*P[0]*V[0]
    return u, R, V, e, P, C, t, V_0, r

def get_q( u, V_1, V__0, vis_a, index):
    # j is the space coordinate of the interest point
    # u is an array with all info about vel for all points at a specific time
    # V_1 is an array with all volume info for all points at a time step
    # after the interest time 
    # V_0 is the array wiht all volume info far all points at the time of interest
    # return a viscousity array from index 0 to n-1 with length n
    # but the useful length is 0 to n-2, that is q[:-1]
    q = np.zeros(len(u)) 
    #for j in range(len(q)-1):
    for j in range(len(q)-1):
        if u[j+1] - u[j] < 0 :
             q[j] = 2.0*vis_a*vis_a*(u[j+1] - u[j])*(u[j+1]-u[j])/(V_1[j]+V__0[j])
    #print u[1] - u[2], (V_1[1]+V__0[2])/2.0, q[1]
    return q

def get_C( C0, V, P,  gamma, index):
    # the P and V are arrays of P and V at time of interst (1D array)
    C = np.sqrt(gamma*P*V)
    return C

def get_dt(R, C, vis_b, ratio_of_critical ):
    # R and C are 1-D array
    R_1 = R[1:]
    R_0 = R[:-1]
    dt_arr = vis_b* (R_1 - R_0)/C[:-1]
    return ratio_of_critical*np.nanmin(dt_arr)

def update_u(dt, u, R, P,  q, V_0, r, alpha, index):
    # the good points are from index 1 to index n-1 
    # the index 0 is at the piston and is not good
    # and index n are not important
    V_arr = 0.5*(V_0[1:-1]+V_0[:-2])            # array of length n -2 
    P_arr = (P[1:-1] - P[:-2])                  # P is array of length n and P_arr of length n-2  1:-1 ,  :-2
    q_arr = (q[1:-1] - q[:-2])                  # q is array of length n and q_arr of length n-2  1:-1 ,  :-2
    r_arr = 0.5*(r[2:] - r[:-2])            # r is array of length n and r-arr of length n-2
    u_arr = u[1:index]
    u_array = np.zeros(len(u))
    u_array[1:-1] = u_arr - dt*V_arr*(R[1:-1]/r[1:-1])**(alpha - 1)*(P_arr+q_arr)/(r_arr)
    u_array[0] = u[0]
    u_array[-1] = u[-1]
    return u_array

def update_R(dt, u , R, index):
    R_new = dt*u + R
    R[-1] = 1000*R[-1]
    return  R_new

def get_V(R, V, V_0, r, index):
    # update 0 - n-1 term since there should be only n-1 half grids
    #V_arr = np.zeros(len(V))
    V_arr = deepcopy(V)
    V_arr[:-1] = V_0[:-1]*(R[1:]**(alpha) - R[:-1]**(alpha))/(r[1:]**(alpha)-r[:-1]**(alpha))
    V_arr[-1] = V[-1]
    return V_arr 

def get_e(V_1, V__0, e, P, q_1, gamma, index):
    e_arr = deepcopy(e)
    arr_1 =  (1.0+ (gamma - 1.0)*0.5*(V_1[:-1]-V__0[:-1])/V_1[:-1])
    arr_2 = e[:index]-(0.5*P[:index]+q_1[:-1])*(V_1[:-1]- V__0[:-1])
    e_arr[:index] = arr_2/arr_1
    e_arr[index] = e[index]
    return e_arr

def get_P(P, e_1,V_1, gamma, index):
    #P_arr = np.zeros(len(P))
    P_arr = deepcopy(P)
    P_arr[:-1] = (gamma-1.0)*e_1[:-1]/V_1[:-1]
    P_arr[-1] = P[-1]
    return P_arr

def get_tot_energy(n, V, P, e, u, q, qwork, E0):
    kinetic = np.sum(0.5*(0.5*(u[0][1:]+u[1][1:]))**2)     # 1
    thermal = np.sum(e[:-1])            
    total = kinetic + thermal 
    qwork = -np.sum((q[:-1])*(V[1][:-1] - V[0][:-1]))
    error = (total-E0)/E0
    return kinetic , thermal, qwork, total, error

def update_all(n,u, R, V, e, P, C, q,  t, V_0, r, alpha, vis_a, vis_b, gamma,dt ,ratio_of_critical, shock_index, n_p):
    
    index = int(shock_index)
    t[1] = t[0] + dt
    u[1] = update_u(dt, u[0], R[0], P[0],  q, V_0, r, alpha, index)
    R[1] = update_R(dt, u[1] , R[0], index)
    V[1] = get_V(R[1],V[0],V_0,r, index)
    q_new  = get_q(u[1], V[1], V[0], vis_a, index)
    e[1] = get_e(V[1],V[0],e[0],P[0],q_new,gamma, index)
    P[1] = get_P(P[0],e[1],V[1],gamma, index)
    C[1] = get_C(C[0], V[1], P[1],gamma, index)
    dt_new = get_dt(R[1], C[1], vis_b, ratio_of_critical)
    
    return q_new, dt_new

def update_array(u, R, V, e, P, C, q,  t):
    t[0] = t[1]
    u[0] = deepcopy(u[1])
    R[0] = deepcopy(R[1])
    V[0] = deepcopy(V[1])
    e[0] = deepcopy(e[1])
    P[0] = deepcopy(P[1])
    C[0] = deepcopy(C[1])
    return 

def get_strong_shock_boundary(U_p, gamma, x_shock, dt ):
    #speed  = (gamma+1.0)/2.0*U_p
    speed = 0.25*(U_p*(gamma + 1.) + np.sqrt(16.0*gamma + U_p*U_p*(1. + gamma)*(1. + gamma)))
    #speed = np.sqrt(V_0/2.0 *( (gamma+1.0)*P+ (gamma -1.0) ))
    return x_shock + speed*dt

def find_error( test_arr, analytic_arr,  up_index, down_index=0):
    analytic_arr = analytic_arr[down_index:up_index]
    analytic_max = np.nanmax(analytic_arr)
    test_arr = test_arr[down_index:up_index]
    diff = test_arr - analytic_arr
    for i in range(len(diff)):
        #divisor = max(abs(test_arr[i]), abs(analytic_arr[i]))
        #if abs(analytic_arr[i]) <= 1.0e-3*analytic_max:
        divisor = 1.0
        diff[i] = diff[i]/divisor
    return diff

def plot_shock(n, u, R, P, V, e, C, r,  x_shock, P_max, time,  savefile, lagrangean = False ):
    
    shock_index = np.argmax(P[1])
    R_list = R[1] #/R[1][shock_index]
    u_list = u[1] #/np.nanmax(u[1])
    x_axis = 0.5*(R_list[1:] + R_list[:-1])
    u_cent = 0.5*(u_list[1:]+u_list[:-1])
    u_prime = 0.5*(u_list[1:]+u_list[:-1])/x_axis
    #analytical = get_array_from_eta(5.0/3.0, x_axis, u_prime)
    x_name = "R/R0"

    if lagrangean:
        x_axis = 4*np.pi*r**(3)/3.0/(shock_index)
        x_name = "mass ratio"
    
    #plt.subplot(121)
    plt.xlabel(x_name)
    plt.plot(R_list[:-2], u_list[:-2]/C[0][1], label = "u/C0" , marker = ".", color = "green" )
    plt.plot(x_axis[:-1], 0.1/ V[1][:-2], marker = ".", label = "rho/rho_0", color = "red")
    plt.plot(x_axis[:-1], P[1][:-2]/1.0, marker = ".", label = "P/P0", color = "blue" )
    plt.suptitle("t = "+ str(time))
    plt.show()
    #plt.savefig(savefile)
    plt.close()
    return 

def parameter_write(time,array,index, fo):
    fo.write(str(time))
    up_lim = len(array)
    for i in range(index):
        #j = min(int((0.2*(i/10.0)+1)*i), up_lim-1)
        fo.write(str(array[i])+'\t')
    fo.write("\n")
    return

def main_evolve(U_p, P_p, gamma, vis_a, vis_b, alpha, n_p, n_t, ratio_of_critical, Plot):
    u, R, V, e, P, C, t, V_0, r = initial(U_p, P_p, n_p, n_t, gamma, alpha)
    q = get_q( u[0], V[0], V[0], vis_a, n_c)
    dt = get_dt(R[0],C[0], vis_b, ratio_of_critical)
    qwork = 0.0
    x_shock = 0.0
    shock_index = n_c
    dt_min = 1.0e-7
    fo = open("initial_rho_0.txt","w")
    E0 = np.sum(0.5*(u[0]*u[0]) + e[0]) 
    
    for n in range(1,n_t-1):
        q,dt = update_all(n,u, R, V, e, P, C, q, t, V_0, r, alpha, vis_a, vis_b, gamma,dt ,ratio_of_critical, n_p-1, n_p)
        kinetic , thermal, qwork,  total, error = get_tot_energy( n, V, P, e[0], u, q, qwork, E0 )
        shock_index = np.argmin(V[1])
        x_shock = R[1][shock_index]
        if n == 0:
            x_shock = r[1]
        time = t[0]
        print time, kinetic , thermal, qwork,  total, x_shock, error
        P_max = np.amax(P[1][1:])
        
	if (n!=0 and (n+1) % 10000 == 0 and Plot):
            #if (n == n_t - 10):
            savefile = "/Users/Chadwick/Documents/my study/Subjects/physics/physics 598/hw_5/strong_shock/"+str(n).zfill(8)+".png"
            #plot_shock(n, u, R, P, V, e, C,  r,  x_shock, P_max, time,  savefile, lagrangean = False)
	#if n %50 ==0 :
        #    parameter_write(time, R[0], 200, fo)
        update_array(u, R, V, e, P, C, q,  t)
    savefile = "something"
    plot_shock(n, u, R, P, V, e, C,  r,  x_shock, P_max, time,  savefile, lagrangean = False)
    return
###### main ###########################
#print get_initial_r(np.zeros(n_p)+1.0 ,alpha,n_p)
main_evolve(U_p, P_p, gamma, vis_a, vis_b, alpha,  n_p, n_t,  ratio_of_critical, Plot = plot)
#u, R, V, e, P, C, t, V_0, r = initial(U_p, P_p, n_p, n_t, gamma, alpha)
#print r[:4]
