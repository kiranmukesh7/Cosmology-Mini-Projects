#############################
#### Importing Libraries ####
#############################

import numpy as np
from astropy import constants as const
from astropy import units as u

#############################
#### Function Definition ####
#############################

E = lambda z: omega0["rad"]*(1+z)**4 + omega0["m"]*(1+z)**3 + omega0["lambda"]


def f(z):
	Z = -1.0/(H0*(1+z)*np.sqrt(E(z)))
	return Z

def integrate():
	a,b=1000,0
	eps = 1e-12
	while(f(a)>eps):
		a *= 10
	n = a*1000
	x = np.linspace(a,b,n)
	s = np.sum(f(x[1:-1]))
	s += 0.5*(f(x[0])+f(x[-1]))
	s *= float(b-a)/(n)
	return s

#############################
######## Calculation ########
#############################

# given:
pc_to_m = const.pc.value
H0 = 69.6 # km/s/Mpc <- assumed value - not given
H0 = H0*(1e3/(1e6*pc_to_m)) 
omega0 = {"rad":  8.4e-5, "m": 0.286, "lambda": 0.714}
s_to_yr = (1*u.second).to(u.yr).value
s_to_Gyr = s_to_yr/1e9
Gyr_to_s = 1.0/s_to_Gyr
t0 = integrate()*s_to_Gyr
print("Present age of universe = {} Gyrs".format(t0))

#############################
######## End Of Code ########
#############################

