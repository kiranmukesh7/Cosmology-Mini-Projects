#############################
#### Importing Libraries ####
#############################

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import matplotlib as mpl
from astropy import constants as const
from scipy import integrate
from scipy import stats
from scipy.optimize import curve_fit
import pandas as pd
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

#############################
#### Function Definition ####
#############################

def visualize(x,y,dy=None,title="data",typ="logx"):
    fs = 15
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    ax.errorbar(x,y,dy,fmt='.',color="k",capsize=4,ecolor="gray",elinewidth=1.3)
    ax.scatter(x,y,s=10)
    if(typ=="logx"):
        ax.set_xscale("log")
    plt.xlabel("Redshift (z)",fontsize=fs)
    plt.ylabel(r"Distance modulus (m-M)",fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.title("Distance Modulus vs Redshift",fontsize=fs+5)
    plt.savefig("{}.png".format(title),bbox_inches="tight")
    plt.close()

def H(z,omega_m,omega_lam):
    return H0*np.sqrt(E(z,omega_m,omega_lam))

def Integrand(z,omega_m,omega_lam):
    return 1.0/H(z,omega_m,omega_lam)

def get_dL(z,omega_m,omega_lam,unit="Mpc"):
    integrand = lambda z: Integrand(z,omega_m=omega_m,omega_lam=omega_lam)
    if(unit == "m"):
        d,dd = integrate.quad(integrand,0,z)
        d *= (c*(1+z))
        dd *= (c*(1+z))
        return [d,dd]
    if(unit == "Mpc"):
        if(isinstance(z, list) or isinstance(z, np.ndarray)):
            d,dd = np.zeros(len(z)),np.zeros(len(z))
            for i in range(len(z)):
                d[i],dd[i] = integrate.quad(integrand,0,z[i])
        else:
            d,dd = integrate.quad(integrand,0,z)
        d *= ((c*m_to_Mpc)*(1+z))
        dd *= ((c*m_to_Mpc)*(1+z))
        return [d,dd]

def reduced_chi_square(x,y,s,m): # ddof = v
    v = x.size - m
    chi2 = (np.sum((x-y)**2/s**2))/v
    p = 1 - stats.chi2.cdf(chi2, v)
    return chi2,p

def E(z,omega_m,omega_lam):
    return omega_m*(1+z)**3 + omega_lam

def dm_to_dL(dm,dmerr):
    dL = 10**((dm-25.0)/5.0) # in Mpc
    dLerr = dL*np.log(10)*(dmerr/5.0)
    return dL,dLerr

def dL_to_dm(dL,dLerr):
    dm = 5*np.log10(dL) + 25
    dmerr = (dLerr*5*np.log(10))/dL
    return dm,dmerr

def get_dm_arr(z_arr,omega_m,omega_lam=None):
    if(omega_lam==None):
        dL = np.array([get_dL(z,omega_m,1-omega_m) for z in z_arr])    
    else:
        dL = np.array([get_dL(z,omega_m,omega_lam) for z in z_arr])    
    dL_arr = dL.T[0]
    ddL_arr = dL.T[1]
    return dL_to_dm(dL_arr,ddL_arr)

def fit_func_unconstrained(z,omega_m,omega_lam):
    dL = get_dL(z,omega_m,omega_lam,unit="Mpc")
    dm,dm_err = dL_to_dm(dL[0],dL[1])
    return dm

def fit_func_constrained(z,omega_m):
    dL = get_dL(z,omega_m,1-omega_m,unit="Mpc")
    dm,dm_err = dL_to_dm(dL[0],dL[1])
    return dm

def fit_func_unconstrained_L(z,omega_m,omega_lam):
    dL = get_dL(z,omega_m,omega_lam,unit="Mpc")
    return dL[0]

def fit_func_constrained_L(z,omega_m):
    dL = get_dL(z,omega_m,1-omega_m,unit="Mpc")
    return dL[0]

def plot_best_fit(m,lam,method="grid_search",typ="normal"):
    fs=20.0
    fig1 = plt.figure(1,figsize=(12,8))
    #Plot Data-model
    frame1 = fig1.add_axes((.1,.3,.8,.6))
    plt.errorbar(data["z"],data["dm"],yerr=data["dmerr"],fmt='.',color="black",capsize=4,ecolor='gray',elinewidth=1.3, label="Type Ia SNe Photometric Data")
    if(typ=="errorbar"):
        plt.errorbar(x=z_arr,y=dm_arr["best"],yerr=ddm_arr["best"],color="darkgreen",capsize=4,ecolor='darkgreen',elinewidth=1.3,label=r"$\Omega_m$={},$\Omega_\Lambda$={}".format(np.round(m,2),np.round(lam,2)))
    else:
        plt.plot(z_arr,dm_arr["best"],color="darkgreen",label=r"$\Omega_m$={},$\Omega_\Lambda$={}".format(np.round(m,2),np.round(lam,2)))
    frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
    plt.xlabel("Redshift (z)",fontsize=fs)
    plt.ylabel(r"Distance modulus (m-M)",fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.title("Distance Modulus vs Redshift",fontsize=fs+5)
    plt.legend(fontsize=fs)
    #Residual plot
    difference = data["dm"] - dm["best"] 
    frame2=fig1.add_axes((.1,.1,.8,.2))        
    plt.errorbar(data["z"],difference,yerr=ddm["best"] + data["dmerr"],fmt='.',color="black",capsize=4,ecolor='darkgreen',elinewidth=1.3)
    plt.ylabel("Residue",fontsize=fs)
    plt.xlabel("Redshift (z)",fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.savefig("best_fit_with_residual_plot_{}_{}.png".format(method,typ),bbox_inches="tight")
    plt.close()

def plot_diff_models(m,lam,method="grid_search",typ="normal",scale="logx"):
    fs=20.0
    fig1 = plt.figure(1,figsize=(12,8))
    plt.errorbar(data["z"],data["dm"],yerr=data["dmerr"],fmt='.',color="black",capsize=4,ecolor='gray',elinewidth=1.3,label="Type Ia SNe Photometric Data")
    if(typ == "errorbar"):
        plt.errorbar(x=z_arr,y=dm_arr["best"],yerr=ddm_arr["best"],color="darkgreen",capsize=4,ecolor='darkgreen',elinewidth=1.3,label=r"$\Omega_m$={}, $\Omega_\Lambda$={} (Best fit model)".format(np.round(m,2),np.round(lam,2) ))
        plt.errorbar(x=z_arr,y=dm_arr["0"],yerr=ddm_arr["0"],color="navy",capsize=4,ecolor='navy',elinewidth=1.3,label=r"$\Omega_m$={}, $\Omega_\Lambda$={}".format(0,1.0))
        plt.errorbar(x=z_arr,y=dm_arr["1"],yerr=ddm_arr["1"],color="crimson",capsize=4,ecolor='crimson',elinewidth=1.3,label=r"$\Omega_m$={}, $\Omega_\Lambda$={}".format(1,0))
        plt.errorbar(x=z_arr,y=dm_arr["0.5"],yerr=ddm_arr["0.5"],color="darkorange",capsize=4,ecolor='darkorange',elinewidth=1.3,label=r"$\Omega_m$={}, $\Omega_\Lambda$={}".format(0.5,0.5))
        plt.errorbar(x=z_arr,y=dm_arr["1.5"],yerr=ddm_arr["1.5"],color="deepskyblue",capsize=4,ecolor='deepskyblue',elinewidth=1.3,label=r"$\Omega_m$={}, $\Omega_\Lambda$={}".format(1.5,-0.5))
    else:
        plt.plot(z_arr,dm_arr["best"],lw=3,color="darkgreen",label=r"$\Omega_m$={}, $\Omega_\Lambda$={} (Best fit model)".format(np.round(m,2),np.round(lam,2) ))
        plt.plot(z_arr,dm_arr["0"],'--',color="navy",label=r"$\Omega_m$={}, $\Omega_\Lambda$={}".format(0,1.0))
        plt.plot(z_arr,dm_arr["1"],'--',color="crimson",label=r"$\Omega_m$={}, $\Omega_\Lambda$={}".format(1,0))
        plt.plot(z_arr,dm_arr["0.5"],'--',color="darkorange",label=r"$\Omega_m$={}, $\Omega_\Lambda$={}".format(0.5,0.5))
        plt.plot(z_arr,dm_arr["1.5"],'--',color="deepskyblue",label=r"$\Omega_m$={}, $\Omega_\Lambda$={}".format(1.5,-0.5))
    if(scale=="logx"):
        plt.xscale("log")
    plt.xlabel("Redshift (z)",fontsize=fs)
    plt.ylabel(r"Distance modulus (m-M)",fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.title("Distance Modulus vs Redshift",fontsize=fs+5)
    plt.legend(fontsize=fs,loc="lower right")
    plt.savefig("different_models_{}_{}_{}.png".format(method,typ,scale),bbox_inches="tight")
    plt.close()
    
def plot_residuals(m,lam,method="grid_search",typ="normal",scale="logx"):
    fs=20.0
    fig1 = plt.figure(1,figsize=(12,8))
    plt.errorbar(data["z"],data["dm"]-dm["best"],yerr=ddm["best"]+data["dmerr"],fmt='o',color="black",capsize=4,ecolor='gray',elinewidth=1.3,label="Type Ia SNe Photometric Data")
    if(typ=="errorbar"):
        plt.errorbar(x=data["z"],y=dm["best"]-dm["best"],yerr=ddm["0"]+ddm["best"],color="darkgreen",capsize=2,ecolor='darkgreen',elinewidth=1.3,label=r"$\Omega_m$={}, $\Omega_\Lambda$={}".format(np.round(m,2),np.round(lam,2)))
        plt.errorbar(x=data["z"],y=dm["0"]-dm["best"],yerr=ddm["0"]+ddm["best"],color="navy",capsize=2,ecolor='navy',elinewidth=1.3,label=r"$\Omega_m$={}, $\Omega_\Lambda$={}".format(0,1.0))
        plt.errorbar(x=data["z"],y=dm["1"]-dm["best"],yerr=ddm["1"]+ddm["best"],color="crimson",capsize=2,ecolor='crimson',elinewidth=1.3,label=r"$\Omega_m$={}, $\Omega_\Lambda$={}".format(1,0))
        plt.errorbar(x=data["z"],y=dm["0.5"]-dm["best"],yerr=ddm["0.5"]+ddm["best"],color="darkorange",capsize=2,ecolor='darkorange',elinewidth=1.3,label=r"$\Omega_m$={}, $\Omega_\Lambda$={}".format(0.5,0.5))
        plt.errorbar(x=data["z"],y=dm["1.5"]-dm["best"],yerr=ddm["1.5"]+ddm["best"],color="deepskyblue",capsize=2,ecolor='deepskyblue',elinewidth=1.3,label=r"$\Omega_m$={}, $\Omega_\Lambda$={}".format(1.5,-0.5))
    else:
        plt.plot(z_arr,dm_arr["best"]-dm_arr["best"],color="darkgreen",label=r"$\Omega_m$={}, $\Omega_\Lambda$={}".format(np.round(m,2),np.round(lam,2)))
        plt.plot(z_arr,dm_arr["0"]-dm_arr["best"],color="navy",label=r"$\Omega_m$={}, $\Omega_\Lambda$={}".format(0,1.0))
        plt.plot(z_arr,dm_arr["1"]-dm_arr["best"],color="crimson",label=r"$\Omega_m$={}, $\Omega_\Lambda$={}".format(1,0))
        plt.plot(z_arr,dm_arr["0.5"]-dm_arr["best"],color="darkorange",label=r"$\Omega_m$={}, $\Omega_\Lambda$={}".format(0.5,0.5))
        plt.plot(z_arr,dm_arr["1.5"]-dm_arr["best"],color="deepskyblue",label=r"$\Omega_m$={}, $\Omega_\Lambda$={}".format(1.5,-0.5))
    if(scale=="logx"):
        plt.xscale("log")
    plt.xlabel("Redshift (z)",fontsize=fs)
    plt.ylabel(r"Residual Distance modulus ($\Delta$(m-M))",fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.title("Distance Modulus vs Redshift",fontsize=fs+5)
    plt.legend(fontsize=fs,loc="best")
    plt.savefig("different_models_residuals_{}_{}_{}.png".format(method,typ,scale),bbox_inches="tight")
    plt.close()

#############################
####### Loading Data ########
#############################

data = []
with open("SCPUnion2.1_mu_vs_z low error.txt") as f:
    for line in f:
        data.append(line.split()[:3])
data = np.array([[float(i) for i in row ] for row in data[1:]])
data = {"z":data.T[0],"dm":data.T[1],"dmerr":data.T[2]}
data["dL"],data["dLerr"] = dm_to_dL(data["dm"],data["dmerr"])

#############################
#### Data Visualization #####
#############################

visualize(data["z"],data["dm"],data["dmerr"],"data_logx")
visualize(data["z"],data["dm"],data["dmerr"],"data","normal")
visualize(data["z"],data["dL"],data["dLerr"],"data_L_logx")
visualize(data["z"],data["dL"],data["dLerr"],"data_L","normal")

#############################
######## Calculation ########
#############################

###############################################################
###--- Method-1: Fitting models to Distance Modulus Data ---###
###############################################################

print("Fitting models to distance modulus data.")
# grid search under the constrain omega_m + omega_lam = 1
c = const.c.value
m_to_pc = 1.0/const.pc.value
m_to_Mpc = (1e-6)/const.pc.value
pc_to_m = const.pc.value
H0 = 69.7 # km/s/Mpc
H0 = H0*(1e3/(1e6*pc_to_m)) 
omega_m = np.arange(0.2,0.4,0.01)
omega_lambda = 1-omega_m
z_arr = np.linspace(np.amin(data["z"]),np.amax(data["z"]),1000)
rchi2_arr = np.zeros(len(omega_m))
p_arr = np.zeros(len(omega_m))
dL_arr = np.zeros(len(z_arr))
ddL_arr = np.zeros(len(z_arr))
for ctr,(i,j) in enumerate(zip(omega_m,omega_lambda)):
    dL = np.array([get_dL(z,i,j) for z in data["z"]])
    dm,dmerr = dL_to_dm(dL.T[0],dL.T[1])
    rchi2,p = reduced_chi_square(dm,data["dm"],data["dmerr"],1)
    rchi2_arr[ctr] = rchi2
    p_arr[ctr] = p

idx = np.argmin(rchi2_arr)
print("Best fit parameter values determined by constrained grid search: \n omega_m = {}, omega_lam = {}".format(np.round(omega_m[idx],2),np.round(1-omega_m[idx],2)))

popt_unc, pcov_unc = curve_fit(fit_func_unconstrained, data["z"], data["dm"], sigma=data["dmerr"],bounds=[0,1])
print("Best fit parameter values determined by scipy's curve-fit function, without any constraint on the values of omega_m and omega_lam: \n omega_m = {}, omega_lam = {}".format(np.round(popt_unc[0],2),np.round(popt_unc[1],2)))

popt_c, pcov_c = curve_fit(fit_func_constrained, data["z"], data["dm"], sigma=data["dmerr"],bounds=[0,1])
print("Best fit parameter values determined by scipy's curve-fit function, under the constraint that omega_m + omega_lam = 1: \n omega_m = {}, omega_lam = {}".format(np.round(popt_c[0],2),np.round(1-popt_c[0],2)))

best_m = [omega_m[idx],popt_c[0],popt_unc[0]]
best_lam = [1-omega_m[idx],1-popt_c[0],popt_unc[1]]

###############################################################
###---  Method-1: Fitting models to Luminosity Distance  ---###
###############################################################
print("Fitting models to luminosity distance data.")
# grid search under the constrain omega_m + omega_lam = 1
# fitting dL
rchi2_arr_L = np.zeros(len(omega_m))
p_arr_L = np.zeros(len(omega_m))
dL_arr = np.zeros(len(z_arr))
ddL_arr = np.zeros(len(z_arr))

for ctr,(i,j) in enumerate(zip(omega_m,omega_lambda)):
    dL = np.array([get_dL(z,i,j) for z in data["z"]])
    rchi2,p = reduced_chi_square(dL.T[0],data["dL"],data["dLerr"],1)
    rchi2_arr_L[ctr] = rchi2
    p_arr_L[ctr] = p

idx = np.argmin(rchi2_arr_L)
print("Best fit parameter values determined by constrained grid search: \n omega_m = {}, omega_lam = {}".format(np.round(omega_m[idx],2),np.round(1-omega_m[idx],2)))

popt_unc_L, pcov_unc_L = curve_fit(fit_func_unconstrained_L, data["z"], data["dm"], sigma=data["dmerr"],bounds=[0,1])
print("Best fit parameter values determined by scipy's curve-fit function, without any constraint on the values of omega_m and omega_lam: \n omega_m = {}, omega_lam = {}".format(np.round(popt_unc_L[0],2),np.round(popt_unc_L[1],2)))

popt_c_L, pcov_c_L = curve_fit(fit_func_constrained_L, data["z"], data["dm"], sigma=data["dmerr"],bounds=[0,1])
print("Best fit parameter values determined by scipy's curve-fit function, under the constraint that omega_m + omega_lam = 1: \n omega_m = {}, omega_lam = {}".format(np.round(popt_c_L[0],2),np.round(1-popt_c_L[0],2)))

best_m_L = [omega_m[idx],popt_c[0],popt_unc[0]]
best_lam_L = [1-omega_m[idx],1-popt_c[0],popt_unc[1]]

#############################
########## Plotting #########
#############################

dm_arr = {}
ddm_arr = {}
dm = {}
ddm = {}

dm_arr["0"],ddm_arr["0"] = get_dm_arr(z_arr,0)
dm_arr["1"],ddm_arr["1"] = get_dm_arr(z_arr,1)
dm_arr["0.5"],ddm_arr["0.5"] = get_dm_arr(z_arr,0.5)
dm_arr["1.5"],ddm_arr["1.5"] = get_dm_arr(z_arr,1.5)

dm["0"],ddm["0"] = get_dm_arr(data["z"],0)
dm["1"],ddm["1"] = get_dm_arr(data["z"],1)
dm["0.5"],ddm["0.5"] = get_dm_arr(data["z"],0.5)
dm["1.5"],ddm["1.5"] = get_dm_arr(data["z"],1.5)

methods = ["grid_search","constrained_curvefit","unconstrained_curvefit"]
for method,m,lam in zip(methods,best_m,best_lam):
    dm_arr["best"],ddm_arr["best"] = get_dm_arr(z_arr,m,lam)
    dm["best"],ddm["best"] = get_dm_arr(data["z"],m,lam)
    plot_best_fit(m,lam,method,"normal")
    plot_best_fit(m,lam,method,"errorbar")
    plot_diff_models(m,lam,method,"normal","normal")
    plot_diff_models(m,lam,method,"errorbar","normal")
    plot_diff_models(m,lam,method,"normal","logx")
    plot_diff_models(m,lam,method,"errorbar","logx")
    plot_residuals(m,lam,method,typ="normal",scale="logx")
    plot_residuals(m,lam,method,typ="normal",scale="normal")
    plot_residuals(m,lam,method,typ="errorbar",scale="logx")
    plot_residuals(m,lam,method,typ="errorbar",scale="normal")

for method,m,lam in zip(methods,best_m_L,best_lam_L):
    dm_arr["best"],ddm_arr["best"] = get_dm_arr(z_arr,m,lam)
    dm["best"],ddm["best"] = get_dm_arr(data["z"],m,lam)
    plot_best_fit(m,lam,"L_" + method,"normal")
    plot_best_fit(m,lam,"L_" + method,"errorbar")
    plot_diff_models(m,lam,"L_" + method,"normal","normal")
    plot_diff_models(m,lam,"L_" + method,"errorbar","normal")
    plot_diff_models(m,lam,"L_" + method,"normal","logx")
    plot_diff_models(m,lam,"L_" + method,"errorbar","logx")
    plot_residuals(m,lam,"L_" + method,typ="normal",scale="logx")
    plot_residuals(m,lam,"L_" + method,typ="normal",scale="normal")
    plot_residuals(m,lam,"L_" + method,typ="errorbar",scale="logx")
    plot_residuals(m,lam,"L_" + method,typ="errorbar",scale="normal")

#############################
######## End Of Code ########
#############################
