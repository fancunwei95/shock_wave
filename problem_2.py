import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton

def v1(gamma):
	return (13.0*gamma*gamma-7.0*gamma+12.0)/(2.0*gamma+1.0)/(3.0*gamma-1.0)

def v2(gamma):
	return 5.0*(gamma-1.0)/(2.0*gamma+1.0)

def v3(gamma):
	return 3.0/(2.0*gamma+1.0)

def v4(gamma):
	return v1(gamma)/(2-gamma)

def v5(gamma):
	return 2.0/(gamma-2.0)

def v6(gamma):
	return (13.0*gamma*gamma-7.0*gamma+12.0)/(3.0*gamma-1.0)/(2.0-gamma)/5.0

def v7(gamma):
	return gamma/(gamma-2.0)

def v_prime(x,eta,gamma):
	bad_part = (gamma-1.0)/(2.0*gamma * x - gamma - 1.0)
	if bad_part < 0.0:
		return 0.0
	return x*x*((5.0*(gamma+1.0)-2*x*(3.0*gamma-1.0))/(7.0-gamma))**(v1(gamma))*(gamma-1.0)**(v2(gamma))*(eta)**5.0 -(2.0*gamma * x - gamma - 1.0)**(v2(gamma))

def not_bad_p(x,eta,gamma):
	bad_part = (gamma-1.0)/(2.0*gamma * x - gamma - 1.0)
	if bad_part < 0.0:
		return False
	return True

def rho_prime(v_p,gamma):
	p1 = (2*gamma*v_p - gamma - 1.0)/(gamma - 1.0)
	p2 = ((5.0*(gamma+1.0)-2*v_p*(3.0*gamma-1.0))/(7.0-gamma))
	p3 = (gamma+1.0-2.0*v_p)/(gamma - 1.0)
	if p1 < 0.0 :
                p1 = 0.0
	return (p1)**(v3(gamma))*(p2)**(v4(gamma))*(p3)**(v5(gamma))

def p_prime_eta2(v_p,gamma):
	p1 = ((5.0*(gamma+1.0)-2*v_p*(3.0*gamma-1.0))/(7.0-gamma))
	p2 = (gamma+1.0-2.0*v_p)/(gamma - 1.0)
	
	return p1**(v6(gamma))*(p2)**(v7(gamma))*v_p**(6.0/5.0)

def temperature(P,rho,eta):
	if rho == 0.0 or eta ==0.0 :
		return 5.0
	return np.log10( P/rho/eta/eta )

def get_m(eta_2P,rho, gamma):
	# use equation mass = tau^(-6/5) and tau = ts/t
	# tau^(6/5) = eta^2(p'/rho'^gamma)
	# derivation is in the homework pdf.
	return rho**(gamma)/eta_2P

def v_with_eta(gamma,Max,d_eta):
	# A 6D  arry saving data required for the output
	# 0: eta, 1: u/u_1, 2:rho/rho_1,  3: p/p_1, 4: log(T/T1), 5: mass/m0
	# using William Newman paper to calculate u/u_1 with 
	# transedental equation solver, here is scipy.optimize.newton
	# use u/u_1 to calculate other terms with equations in the paper
	# then conver eta to mass (eulerian to lagranean)

	num = int(round((Max-d_eta)/(d_eta)))
	v_arr = np.zeros((6,num))    
	eta = Max
	solution = 1.0
	for i in range(num):
		v_arr[0][i] = eta
		if eta > 1.0 :
			continue
		if eta == 1.0 :
			v_arr[1][i] = 1.0 #u/u_1
			v_arr[2][i] = 1.0 #rho/rho_1
			v_arr[3][i] = 1.0 #p/p_1
			v_arr[4][i] = 0.0 #log(T/T1)
			v_arr[5][i] = 1.0 #mass/m1
			eta = eta - d_eta
			continue
		#print eta
		#print solution
		#print v_prime(solution ,eta+d_eta,gamma)
		if (not_bad_p(solution,eta,gamma)):
			# if the bad part is negative, means the solution is off, then stop solve
			# until the solution is back to near the correct the value
			solution = newton(v_prime, solution, None, (eta,gamma), 1.0e-14)  # solution is u'
		v_arr[1][i] = solution * eta  # (2.0)/(gamma+1.0)
		v_arr[2][i] = rho_prime(solution, gamma)
		v_arr[3][i] = p_prime_eta2(solution,gamma)
		v_arr[4][i] = temperature(v_arr[3][i],v_arr[2][i], v_arr[0][i])    # P/rho = T
		v_arr[5][i] = get_m(v_arr[3][i],v_arr[2][i],gamma)
		eta = eta -d_eta
	v_arr = np.fliplr(v_arr)
	return v_arr

def get_array_from_eta(gamma,eta_list, u_list):

	num = len(eta_list)
	v_arr = np.zeros((6,num))    
	solution = 1.0
        for i in range(num):
                eta = eta_list[num-i-1]
		v_arr[0][i] = eta
                if eta > 1.0 :
			continue
		if eta == 1.0 :
			v_arr[1][i] = 1.0 #u/u_1
			v_arr[2][i] = 1.0 #rho/rho_1
			v_arr[3][i] = 1.0 #p/p_1
			v_arr[4][i] = 0.0 #log(T/T1)
			v_arr[5][i] = 1.0 #mass/m1
			continue
		if (not_bad_p(solution,eta,gamma)):
			solution = newton(v_prime, solution, None, (eta,gamma), 1.0e-14)  # solution is u'
		v_arr[1][i] = solution * eta  # (2.0)/(gamma+1.0)
		v_arr[2][i] = rho_prime(solution, gamma)
		v_arr[3][i] = p_prime_eta2(solution,gamma)
		v_arr[4][i] = temperature(v_arr[3][i],v_arr[2][i], v_arr[0][i])
		v_arr[5][i] = get_m(v_arr[3][i],v_arr[2][i],gamma)
        v_arr = np.fliplr(v_arr)
        return v_arr

def find_si(arr, gamma):
	d_eta = arr[0][1]-arr[0][0]
	result = 0.0
	for i in range(len(arr[0])-1):
		eta = 0.5*(arr[0][i] + arr[0][i+1])
		itg_1 = (arr[2][i]*arr[1][i]*arr[1][i])+arr[3][i]
		itg_2 = (arr[2][i+1]*arr[1][i+1]*arr[1][i+1])+arr[3][i+1]
		itg = 0.5*(itg_1+itg_2)
		result = result +  itg*eta*eta*d_eta
	return (25.0*(gamma*gamma-1.0)/(32.0*np.pi)/result)**(0.2)

def plot_arr(arr,show = True):
	plt.subplot(121)
	plt.plot(arr[0],arr[1],"g", label = "v/v1")
	plt.plot(arr[0],arr[2],"r", label = "rho2/rho1")
	plt.plot(arr[0],arr[3],"b", label = "P/P1")
	plt.plot(arr[0],arr[4],"black", label = "log(T/T1)")
	#plt.plot(arr[0],arr[5],"b", label="mass")
	plt.xlabel("eta")
	plt.ylabel("value")
	plt.legend()
	
	plt.subplot(122)
	plt.plot(arr[5],arr[1],"g", label = "v/v1")
	plt.plot(arr[5],arr[2],"r", label = "rho2/rho1")
	plt.plot(arr[5],arr[3],"b", label = "P/P1")
	plt.plot(arr[5],arr[4],"black", label = "log(T/T1)")
	plt.xlabel("mass fraction")
	plt.ylabel("value")
	plt.legend()
	
	if show:
		plt.show()
	return

######## mian ###########################
#gamma = 5.0/3.0
#v_arr = v_with_eta(gamma,1.0,0.005)
#plot_arr(v_arr)
#print "si_0:\t",find_si(v_arr,gamma)


	

		

