#############################
#### Importing Libraries ####
#############################

import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
from scipy import integrate
from astropy import units as u
plt.style.use('dark_background')

#############################
#### Function Definition ####
#############################

E = lambda z: omega0["rad"]*(1+z)**4 + omega0["m"]*(1+z)**3 + omega0["lambda"]

def rk4(x,y0,f,n=100,*args): # n = number of steps between x0 and x
    y = np.zeros(len(x))
    y[0] = y0
    for j in range(len(y)-1):
        h = (x[j+1]-x[j])
        for i in range(n):
            f0 = f( x[j],y[j],*args)
            f1 = f( x[j]+(0.5*h), y[j] + (0.5*h*f0) ,*args)
            f2 = f( x[j]+(0.5*h), y[j] + (0.5*h*f1) ,*args)
            f3 = f( x[j]+h, y[j] + h*f2 ,*args)
            y[j+1] = y[j] + (h/6.0)*(f0 + 2*f1 + 2*f2 + f3)
    return y[::-1]

def z_integrand(z):
    Z = -1.0/(H0*(1+z)*np.sqrt(E(z)))
    return Z

def modified_z_integrand(z):
    Z = -s_to_Gyr/(H0*(1+z)*np.sqrt(E(z)))
    return Z

dzdt = lambda t,z: (-1.0)/z_integrand(z)

#############################
######## Calculation ########
#############################

# given:
pc_to_m = const.pc.value
H0 = 69.7 # km/s/Mpc <- assumed value - not given
print("Assumed value of present Hubble constant = {} km/s/Mpc".format(H0))
H0 = H0*(1e3/(1e6*pc_to_m)) 
omega0 = {"rad":  8.4e-5, "m": 0.27, "lambda": 0.73}
# assuming a0 = 1
G = const.G.value
rho_c0 = (3*(H0**2))/(8*np.pi*G)
print("Present value of critical density = {} km/m^3".format(rho_c0))
rho0 = dict(zip(list(omega0.keys()),[x*rho_c0 for x in list(omega0.values())]))

s_to_yr = (1*u.second).to(u.yr).value
s_to_Gyr = s_to_yr/1e9
Gyr_to_s = 1.0/s_to_Gyr
t0,dt0 = integrate.quad(z_integrand, 0, np.inf)
t0 *= -(s_to_yr/1e9)
dt0 *= -(s_to_yr/1e9)
print("Present age of universe = {} Gyrs".format(t0))

# finding the boundary in terms of z for matter and radiation dominated universes
print("Computing time and redshift at the transition from radiation dominated to matter dominated and matter dominated to dark energy dominated universe.\n")
print("Computing transition time by first determining the transistion redshift and then determining the transition time using integral relation between redshift and time.\n")
z_c = (rho0["m"]/rho0["rad"])-1.0
z_d = np.cbrt(rho0["lambda"]/rho0["m"])-1.0
print("Transition redshift (radiation to matter) = {}".format(np.round(z_c,2)))
print("Transition redshift (matter to dark energy) = {}".format(np.round(z_d,2)))
z_arr = np.logspace(-2,5,1000)
z_arr = np.append([0,z_c,z_d],z_arr)
z_arr = np.sort(z_arr)
t_arr = np.array([integrate.quad(modified_z_integrand,0,z)[0] for z in z_arr]) + t0
# determining the time in the history of universe when z = z_c -- by integration
t_c,dt_c = integrate.quad(modified_z_integrand,0,z_c)
t_c = t0 + t_c
dt_c = dt0 + dt_c
t_d,dt_d = integrate.quad(modified_z_integrand,0,z_d)
t_d = t0 + t_d
dt_d = dt0 + dt_d
print("Transition time (radiation to matter) = {} Yrs".format(np.round(t_c*1e9,2)))
print("Transition time (matter to dark energy) = {} GYrs\n".format(np.round(t_d,2)))

# finding the boundary in terms of t for matter and radiation dominated universes
print("Computing transition redshift by first determining the transistion time by continuity condition on simplified density function and then determining the transition redshift using time-stepping methods.\n")
t_c_new = t0*(rho0["rad"]/rho0["m"])**1.5
idx = np.where(t_arr > t_c_new)
z_c_new = rk4([t_c_new,t_arr[idx][-1]],z_arr[idx][-1],dzdt,1000)[1]
t_d_new = t0*np.sqrt(rho0["m"]/rho0["lambda"])
idx = np.where(t_arr > t_d_new)
z_d_new = rk4([t_d_new,t_arr[idx][-1]],z_arr[idx][-1],dzdt,1000)[1]
print("Transition redshift (radiation to matter) = {}".format(np.round(z_c_new,2)))
print("Transition redshift (matter to dark energy) = {}".format(np.round(z_d_new,2)))
print("Transition time (radiation to matter) = {} Yrs".format(np.round(t_c_new*1e9,2)))
print("Transition time (matter to dark energy) = {} GYrs\n".format(np.round(t_d_new,2)))

z_str = []
for i in z_arr:
    if(i < 1.0):
        z_str.append("{}".format(np.round(i,2)))
    else:
        z_str.append("{:.2e}".format(i))

f = {"rad": lambda z: rho0["rad"]*(1+z)**4,
"m": lambda z: rho0["m"]*(1+z)**(3.0),
"lambda": lambda z: rho0["lambda"],
}

rho = {}
for i in rho0.keys():
    rho[i] = [f[i](z) for z in z_arr]
    
f_new = {"rad": lambda t: np.where(t>t_c_new,rho0["rad"]*(t0/t)**(8.0/3.0),(rho0["rad"]*(t0/t_c_new)**(2.0/3.0))*(t0/t)**2),
"m": lambda t: np.where(t>t_c_new,rho0["m"]*(t0/t)**2,(rho0["m"]*np.sqrt(t0/t_c_new))*(t0/t)**(3.0/2.0)),
"lambda": lambda t: rho0["lambda"],
}

rho_new = {}
for i in rho0.keys():
    rho_new[i] = [f_new[i](t) for t in t_arr]

#############################
########## Plotting #########
#############################

# plotting with the transition redshift determined by first determining the transistion redshift by equating the matter and radiation density and then determining the transition time using integral relation between redshift and time
fs = 15
clrs = ['orange','greenyellow','cyan']
label = ["Radiation Density","Matter Density","Dark Energy Density"]
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(111)
for i,j in enumerate(rho.keys()):
    ax1.plot(np.log10(t_arr),np.log10(rho[j]),label=label[i],c=clrs[i],lw=3)
ax1.axvline(x=np.log10(t_c), label=r'Tranisition time, $t_{\gamma,m}$' + ' = {:.2e} Gyrs'.format(t_c),c='yellow',ls='--')
ax1.axvline(x=np.log10(t_d), label=r'Tranisition time, $t_{m,\Lambda}$' + ' = {} Gyrs'.format(np.round(t_d,2)),c='white',ls='--')
ax1.axvline(x=np.log10(t_arr[0]), label=r'Present time, $t_0$ = {} Gyrs'.format(np.round(t_arr[0],2)),c='pink',ls='--')
plt.legend(loc="best",fontsize=fs)
ax1.set_xlabel("Time since Big Bang, t (in Gyr)",fontsize=fs)
ax1.set_ylabel(r"Density $\rho$ (in $kg/m^{3}$)",fontsize=fs)
ax1.set_title("Evolution of density with cosmic time",fontsize=fs+5)
ax1.tick_params(axis='both', which='major', labelsize=fs)
new_tick_labels = [r"$10^{%s}$" % str(int(lbl)) for lbl in ax1.get_xticks()]
ax1.set_xticklabels(new_tick_labels)
new_tick_labels = [r"$10^{%s}$" % str(int(lbl)) for lbl in ax1.get_yticks()]
ax1.set_yticklabels(new_tick_labels)
ax2 = ax1.twiny()
ax1.grid(True)
new_tick_locations = np.array([t_arr[0],t_c,t_arr[-1]])
new_tick_locations = np.log10(new_tick_locations)
ax2.set_xticks(new_tick_locations)
new_xtick_labels = [z_str[0],"{:.2e}".format(z_c),z_str[-1]]
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticklabels(new_xtick_labels)
ax2.tick_params(axis='both', which='major', labelsize=fs)
ax2.tick_params(axis='both', which='minor', labelsize=fs)
ax2.set_xlabel("Redshift, z",fontsize=fs)
plt.savefig("integral.png",bbox_inches="tight")
plt.close()

# plotting with the transition redshift determined by first determining the transistion time by continuity condition on simplified density function and then determining the transition redshift using time-stepping methods
fs = 15
clrs = ['orange','greenyellow','cyan']
label = ["Radiation Density","Matter Density","Dark Energy Density"]
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(111)
for i,j in enumerate(rho.keys()):
    ax1.plot(np.log10(t_arr),np.log10(rho_new[j]),label=label[i],c=clrs[i],lw=3)
ax1.axvline(x=np.log10(t_c_new), label=r'Tranisition time, $t_{\gamma,m}$' + ' = {:.2e} Gyrs'.format(t_c_new),c='yellow',ls='--')
ax1.axvline(x=np.log10(t_d_new), label=r'Tranisition time, $t_{m,\Lambda}$' + ' = {} Gyrs'.format(np.round(t_d_new,2)),c='white',ls='--')
ax1.axvline(x=np.log10(t_arr[0]), label=r'Present time, $t_0$ = {} Gyrs'.format(np.round(t_arr[0],2)),c='pink',ls='--')
plt.legend(loc="best",fontsize=fs)
ax1.set_xlabel("Time since Big Bang, t (in Gyr)",fontsize=fs)
ax1.set_ylabel(r"Density $\rho$ (in $kg/m^{3}$)",fontsize=fs)
ax1.set_title("Evolution of density with cosmic time",fontsize=fs+5)
ax1.tick_params(axis='both', which='major', labelsize=fs)
new_tick_labels = [r"$10^{%s}$" % str(int(lbl)) for lbl in ax1.get_xticks()]
ax1.set_xticklabels(new_tick_labels)
new_tick_labels = [r"$10^{%s}$" % str(int(lbl)) for lbl in ax1.get_yticks()]
ax1.set_yticklabels(new_tick_labels)
ax2 = ax1.twiny()
ax1.grid(True)
new_tick_locations = np.array([t_arr[0],t_c_new,t_arr[-1]])
new_tick_locations = np.log10(new_tick_locations)
ax2.set_xticks(new_tick_locations)
new_xtick_labels = [z_str[0],"{:.2e}".format(z_c_new),z_str[-1]]
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticklabels(new_xtick_labels)
ax2.tick_params(axis='both', which='major', labelsize=fs)
ax2.tick_params(axis='both', which='minor', labelsize=fs)
ax2.set_xlabel("Redshift, z",fontsize=fs)
plt.savefig("simplified.png",bbox_inches="tight")
plt.close()

#############################
######## End Of Code ########
#############################

