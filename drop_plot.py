import math
from math import cos, sqrt, pi, exp, log
import cmath
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from matplotlib import rcParams
from decimal import Decimal

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ["Garamond"]
rcParams['font.size'] = '20'

# starting with just fitting the through port first
# assumptions: r^2 + k^2 = 1 (unity) --> r as self-coupling and k as cross-coupling

# fitting parameters:
    # p[0] = alpha --> coupling loss
    # p[1] = k
    # p[2] = x axis offset, in units of radians, wavel7ength shift
    # p[3] = y scaling (linear loss)
    # p[4] = y offset

# larger alpha should give wider dip
# smaller kappa should give smaller extinction ratio

p_0 = [0.45, 0.42, 1.2, 0.00037, 0.000, 3.6500, 0.000022, 110e9, -1.2, 0]

# some parameters
ng = 3.5
neff = 3.65
r = 2.4e-3 # radius
L_rt = 2*pi*r

#toggles test plots
test = 1

wavelength, loss =  wavelength_drop, loss_drop = np.loadtxt("C:\\Users\\HopeLee\\Documents\\Data\\Ring Resonators\\LSRL THERM2 W3\\Top_Right\\4C bottom\\Fine Scans\\4C_fine_1543_0_max_drop.csv", delimiter=",", unpack=True, skiprows=19)
a = 1543
b = 1543.2

loss = np.array([10**(-i/10) for i in loss])
wavelength = [i*10**9 for i in wavelength]
phi = [(2*pi/(i*10**-9))*neff*L_rt for i in wavelength]

dwavelength = 1e-3*10**-9
dloss = 10**(-8)

if test == 1:
    plt.figure(figsize=(12,8))
    plt.plot(wavelength, loss, marker=".")
    plt.title('3E W8 Drop')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Insertion Loss')
    # plt.savefig("3E_W9_through_min_plot_only.svg")
    plt.show()

y_max = 0.0023
y_min = 0.0016

#wavelength range of interest for fit

wavelength_range = []
loss_range = []
phi_range = []

for i in wavelength:
    if i >= a and i <= b:
        wavelength_range.append(i)
        loss_range.append(float(loss[np.where(wavelength==i)]))
        phi_range.append(phi[wavelength.index(i)])

# print(phi_range)
range_min = 0
range_max = len(wavelength_range)-1

#test plot
if test ==1 :
    plt.figure(figsize=(12,8))
    plt.errorbar(wavelength_range, loss_range, yerr=dloss, marker=".")
    plt.title('3E W8 Drop')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Insertion Loss')
    plt.show()

def plotting_drop(p, lamb):
    '''lamb taken to be a list of wavelengths for the sweep region'''

    lamb = [i*10**-9 for i in lamb]

    alpha_wg = -log(10**(-p[0]/10))
    a = sqrt(math.exp(-alpha_wg*L_rt))
    A = exp(-alpha_wg*100*L_rt)

    k_1 = p[1]
    k_2 = p[1]

    t1=sqrt(1-k_1**2)
    t2=sqrt(1-k_2**2)

    phi = [(2*pi/i)*neff*L_rt for i in lamb]
    # print(phi[0:5])
    # p[2] = 1j*(2*pi/p[2])*neff*L_rt

    Edrop = [-(k_1)*k_2*sqrt(sqrt(A))*cmath.exp(1j*(i+p[2])/2)/(1-sqrt(A)*(t1)*(t2)*cmath.exp(1j*(i+p[2]))) for i in phi]
    Edrop_sq = [i*i.conjugate() for i in Edrop]
    T_drop = [p[3]*i.real+p[4] for i in Edrop_sq]

    return T_drop

if test == 1:
    y_1 = plotting_drop(p_0, wavelength_range)
    plt.figure(figsize=(12,8))
    plt.errorbar(wavelength_range, loss_range, yerr=dloss, marker=".", label="Data")
    plt.plot(wavelength_range, y_1, label="Guessed Parameters")
    plt.title('4C W9 Drop, Max Model Fit')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Insertion Loss')
    plt.axvspan(wavelength_range[range_min], wavelength_range[range_max], alpha=0.5, color='grey')
    plt.legend(loc="lower right", borderaxespad=0)
    plt.show()

plt.plot(wavelength, loss)
plt.plot(wavelength, plotting_drop(p_0, wavelength))
plt.show()
