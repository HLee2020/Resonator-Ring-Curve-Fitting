import math
import cmath
from math import cos, sqrt, pi, log
import cmath
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ["Garamond"]
rcParams['font.size'] = '20'

test = 1

# fitting parameters:
    # p[0] = alpha --> coupling loss
    # p[1] = k
    # p[2] = x axis offset, in units of radians
    # p[3] = y scaling
    # p[4] = y offset

# larger alpha should give wider dip
# smaller kappa should give smaller extinction ratio

ng = 3.5
neff = 3.65

r = 2.4e-3 # radius
L_rt = 2*pi*r

wavelength, loss = np.loadtxt("C:\\Users\\HopeLee\\Documents\\Data\\Ring Resonators\\LSRL THERM2 W3\\Top_Right\\4C bottom\\Fine Scans\\4C_fine_1543_0_max_through.csv", delimiter=",", unpack=True, skiprows=19)

p0 = [0.4, 0.42, -0.55, 0.00037, 0.0001, 3.6500, 0.000022, 110e9, -1.2, 0]

# if no background, set all sinusoidal parameters to 0!

# p[0] = alpha --> coupling loss
# p[1] = k
# p[2] = x axis offset, in units of radians, wavelength shift
# p[3] = y scaling (linear loss)
# p[4] = y offset
# p[5] = effective n
# p[6] = amplitude of sinusoidal
# p[7] = frequency of sinusoidal
# p[8] = x shift of sinusoidal
# p[9] = y shift of sinusoidal

background_shift = 0.0105

loss = np.array([10**(-i/10) for i in loss])
wavelength = [i*10**9 for i in wavelength]

dwavelength = 1e-3
dloss = 10**(-8)

a = 1543.0
b = 1543.2

y_max = 0.0023
y_min = 0.0016

#wavelength range of interest for fit

wavelength_range = []
loss_range = []

for i in wavelength:
    if i >= a and i <= b:
        wavelength_range.append(i)
        loss_range.append(float(loss[np.where(wavelength==i)]))
#
# plt.figure(figsize=(12,8))
# plt.errorbar(wavelength_range, loss_range, yerr=dloss, marker=".")
# plt.title('3E W8 Through')
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Insertion Loss')
# plt.show()

def through_background (p, x):
    '''lamb taken to be a list of wavelengths for the sweep region'''

    x = [i*10**-9 for i in x]

    alpha_wg = -log(10**(-p[0]/10))
    a = sqrt(math.exp(-alpha_wg*L_rt))
    A = math.exp(-alpha_wg*100*L_rt)

    k_1 = p[1]
    k_2 = p[1]

    t1=sqrt(1-k_1**2)
    t2=sqrt(1-k_2**2);

    phi = [(2*pi/i)*neff*L_rt for i in x]
    reduce_phi = [i%(2*pi) for i in phi]

    Ethru = [(t1-t2*sqrt(A)*np.exp(1j*(i+p[2])))/(1-sqrt(A)*t1*t2*np.exp(1j*(i+p[2]))) for i in reduce_phi]
    Ethru_sq = [i*i.conjugate() for i in Ethru]
    Tthru = [p[3]*i.real+p[4] for i in Ethru_sq]
    Ttotal = [i+p[6]*(np.sin(p[7]*j-p[8]))+p[9] for i,j in zip(Tthru, x)]
    return Ttotal

def through (p, x):
    '''lamb taken to be a list of wavelengths for the sweep region'''

    x = [i*10**-9 for i in x]

    alpha_wg = -log(10**(-p[0]/10))
    a = sqrt(math.exp(-alpha_wg*L_rt))
    A = math.exp(-alpha_wg*100*L_rt)

    k_1 = p[1]
    k_2 = p[1]

    t1=sqrt(1-k_1**2)
    t2=sqrt(1-k_2**2);

    phi = [(2*pi/i)*neff*L_rt for i in x]
    reduce_phi = [i%(2*pi) for i in phi]

    Ethru = [(t1-t2*sqrt(A)*np.exp(1j*(i+p[2])))/(1-sqrt(A)*t1*t2*np.exp(1j*(i+p[2]))) for i in reduce_phi]
    Ethru_sq = [i*i.conjugate() for i in Ethru]
    Tthru = [p[3]*i.real+p[4] for i in Ethru_sq]

    return Tthru

def sinusoidal(p, x):
    x = [i*10**-9 for i in x]

    alpha_wg = -log(10**(-p[0]/10))
    a = sqrt(math.exp(-alpha_wg*L_rt))
    A = math.exp(-alpha_wg*100*L_rt)

    k_1 = p[1]
    k_2 = p[1]

    t1=sqrt(1-k_1**2)
    t2=sqrt(1-k_2**2);

    phi = [(2*pi/i)*neff*L_rt for i in x]
    reduce_phi = [i%(2*pi) for i in phi]

    return [p[6]*(np.sin(p[7]*i-p[8]))+p[9] for i in x]

thru = through(p0, wavelength)
total = through_background(p0, wavelength)
background = sinusoidal(p0, wavelength)
# background_1 = sinusoidal(p1, wavelength)

loss_cleaned = [i-j for i,j in zip(loss, background)]

# plt.figure(figsize=(12,8))
# plt.errorbar(wavelength_range, loss_range, yerr=dloss, marker=".", label="data")
# plt.plot(wavelength_range, thru, label="through plot")
# plt.plot(wavelength_range, back, label="background plot")
# plt.plot(wavelength_range, total, label="total plot")
# plt.title('3E W8 Through')
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Insertion Loss')
# plt.legend()
# plt.show()

plt.figure(figsize=(12,8))
plt.plot(wavelength, total)
plt.plot(wavelength, background, alpha=0.5)
plt.plot(wavelength, loss)
plt.plot(wavelength, thru, alpha=0.5)
plt.show()

# background_shifted = [i+background_shift for i in background]
# plt.figure(figsize=(12,8))
# plt.plot(wavelength, background_shifted)
# plt.plot(wavelength, loss)
# plt.show()

# plt.figure(figsize=(12,8))
# plt.plot(wavelength, loss_cleaned)
# plt.show()
