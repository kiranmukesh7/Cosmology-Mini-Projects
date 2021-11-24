#############################
#### Importing Libraries ####
#############################

import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
from scipy import integrate
plt.style.use('dark_background')

#############################
#### Function Definition ####
#############################

E = lambda z,omega_m,omega_lam: omega_m*(1+z)**3 + omega_lam

def Integrand(z,omega_m,omega_lam):
    return 1.0/(H0*np.sqrt(E(z,omega_m,omega_lam)))

def get_Da(z,omega_m,omega_lam,unit="Mpc"):
    integrand = lambda z: Integrand(z,omega_m=omega_m,omega_lam=omega_lam)
    if(unit == "m"):
        d,dd = integrate.quad(integrand,0,z)
        d *= (c/(1+z))
        dd *= (c/(1+z))
        return [d,dd]
    if(unit == "Mpc"):
        d,dd = integrate.quad(integrand,0,z)
        d *= ((c*m_to_Mpc)/(1+z))
        dd *= ((c*m_to_Mpc)/(1+z))
        return [d,dd]

def plot(x,y,dy=None,typ="semilogy",title="da"):
    fs = 15
    clrs = ['crimson','yellow','cyan','greenyellow','orange']
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    if(typ=="semilogy"):
        for ctr,(i,j) in enumerate(zip(omega_m,omega_lambda)):
            ax.semilogy(x,y[ctr],label=r"$\Omega_m = {}, \Omega_\Lambda = {}$".format(i,j),c=clrs[ctr],lw=3)
    if(typ=="loglog"):
        for ctr,(i,j) in enumerate(zip(omega_m,omega_lambda)):
            ax.loglog(x,y[ctr],label=r"$\Omega_m = {}, \Omega_\Lambda = {}$".format(i,j),c=clrs[ctr],lw=3)
    if(typ=="errorbar"):
        for ctr,(i,j) in enumerate(zip(omega_m,omega_lambda)):
            ax.errorbar(x,y[ctr],dy[ctr],label=r"$\Omega_m = {}, \Omega_\Lambda = {}$".format(i,j),c=clrs[ctr],lw=3)
            
    plt.legend(loc="best",fontsize=fs)
    ax.set_xlabel("Redshift, z",fontsize=fs)
    if(title=="da"):
        ax.set_ylabel(r"Angular diameter distance $d_a$ (in Mpc)",fontsize=fs)
        ax.set_title("Angular diameter distance vs. Redshift",fontsize=fs+5)
    if(title=="dl"):
        ax.set_ylabel(r"Luminosity distance $d_L$ (in Mpc)",fontsize=fs)
        ax.set_title("Luminosity distance vs. Redshift",fontsize=fs+5)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.grid(True)
    plt.savefig("{}_{}.png".format(title,typ),bbox_inches="tight")
    plt.close()

#############################
######## Calculation ########
#############################

# given:
c = const.c.value
m_to_pc = 1.0/const.pc.value
m_to_Mpc = (1e-6)/const.pc.value
pc_to_m = const.pc.value
H0 = 69.7 # km/s/Mpc
H0 = H0*(1e3/(1e6*pc_to_m)) 
omega_m = np.array([1.0,0.7,0.5,0.3,0.0])
omega_lambda = np.copy(omega_m[::-1])

# angular diameter distance
z_arr = np.linspace(0,15,1000)
Da_arr = np.zeros((5,len(z_arr)))
dDa_arr = np.zeros((5,len(z_arr)))
for ctr,(i,j) in enumerate(zip(omega_m,omega_lambda)):
    Da = np.array([get_Da(z,i,j) for z in z_arr])
    Da_arr[ctr] = Da.T[0]
    dDa_arr[ctr] = Da.T[1]

# luminosty distance
Dl_arr = np.zeros((5,len(z_arr)))
dDl_arr = np.zeros((5,len(z_arr)))
for ctr,(i,j) in enumerate(zip(omega_m,omega_lambda)):
    Dl_arr[ctr] = ((1+z_arr)**2)*np.copy(Da_arr[ctr])
    dDl_arr[ctr] = ((1+z_arr)**2)*np.copy(dDa_arr[ctr])

#############################
########## Plotting #########
#############################

plot(x=z_arr,y=Da_arr,typ="semilogy",title="da")
plot(x=z_arr,y=Da_arr,typ="loglog",title="da")
plot(x=z_arr,y=Da_arr,dy=dDa_arr,typ="errorbar",title="da")
plot(x=z_arr,y=Dl_arr,typ="semilogy",title="dl")
plot(x=z_arr,y=Dl_arr,typ="loglog",title="dl")
plot(x=z_arr,y=Dl_arr,dy=dDl_arr,typ="errorbar",title="dl")

#############################
######## End Of Code ########
#############################
