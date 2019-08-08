import math
from math import cos, sqrt, pi, exp, log
import cmath
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import optimize, signal
from matplotlib import rcParams
from decimal import Decimal
import statistics
from statistics import mean
from lmfit import minimize, Parameters, report_fit

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ["Garamond"]
rcParams['font.size'] = '20'

# Loading proper files
wavelength_through, loss_through = np.loadtxt("C:\\Users\\HopeLee\\Documents\\Data\\Ring Resonators\\LSRL THERM2 W3\\Top_Right\\3E bottom\\Fine Scans\\3E_fine_1553_0_max_through.csv", delimiter=",", unpack=True, skiprows=19)
wavelength_drop, loss_drop = np.loadtxt("C:\\Users\\HopeLee\\Documents\\Data\\Ring Resonators\\LSRL THERM2 W3\\Top_Right\\3E bottom\\Fine Scans\\3E_fine_1553_0_max_drop.csv", delimiter=",", unpack=True, skiprows=19)

title = "3E Max Top Right"
f= open(title+'.txt',"w+")

dloss_through = 10**(-4)
dloss_drop = 10**(-4)
f.write("dloss through: "+str(dloss_through)+"\n")
f.write("dloss drop: "+str(dloss_drop)+"\n")

R = 2.4*10**-3
L_rt = 2*pi*R #round trip lengths

# Toggles (1=yes, 0=no)
test = 1
Q_all = 1
save = 1
full_only = 0
manual_through_toggle = 0
manual_drop_toggle = 0

drop_manual = [[1547.81, 1547.825], [1547.985, 1547.995]]

through_manual = [[1547.81, 1547.816], [1547.984, 1547.99]]

# Parameters for through peak finding
prominence_through = 0.00001 # filter to select correct peaks
scale_through = 1.0 # scaling for width of wavelength ranges
shift_through = 0.1 # vertical shift to change dips to peaks to use function

prominence_drop = 0.0002
scale_drop = 1.0

shift_constant = 0
fitting_shift = round((wavelength_through[2]-wavelength_through[1])*10**9*shift_constant, 4)

# Fitting parameters:
p0_t = [0.4, 0.45, -1.8, 0.00025, 0.00015, 3.6500, 0.000033, 110e9, -0.12, 0]
p0_d = [0.4, 0.45, 3.2, 0.0006, 0.00005, 3.6500]

f.write("Through peak finding: \n")
f.write("   Prominence through: "+str(prominence_through)+"\n")
f.write("   Scale through: "+str(scale_through)+"\n")
f.write("   Shift through: "+str(shift_through)+"\n")

f.write("Drop peak finding: \n")
f.write("   Prominence drop: "+str(prominence_drop)+"\n")
f.write("   Scale drop: "+str(scale_drop)+"\n")

f.write("Shift Constant: "+ str(shift_constant)+"\n")
f.write("   Fitting Shift: "+str(fitting_shift)+"\n")
f.write("\n")

if manual_through_toggle == 1:
    f.write("Manual through ranges:\n")
    f.write(str(through_manual)+"\n")
    f.write("\n")

if manual_drop_toggle == 1:
    f.write("Manual drop ranges:\n")
    f.write(str(drop_manual)+"\n")
    f.write("\n")

f.write("Through initial values: "+str(p0_t)+"\n")
f.write("Drop initial values: "+str(p0_d)+"\n")
f.write("\n")

# if no background, use non-background fitting script!!

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

# Setting up lists of data values, round to 4 decimal places for ease of calculations
loss_through = np.array([10**(-i/10) for i in loss_through])
loss_through_shift = np.array([shift_through+(10**(-i/10)) for i in loss_through])  #shift dips to peaks so python function works, only needed for through
wavelength_through = [i*10**9 for i in wavelength_through]
wavelength_through = np.array([round(i, 4) for i in wavelength_through])

loss_drop = np.array([10**(-i/10) for i in loss_drop])
wavelength_drop = [i*10**9 for i in wavelength_drop]
wavelength_drop = np.array([round(i, 4) for i in wavelength_drop])

def resonance_ranges(x_list, y_list, y_list_shift, prom, scale):

    peaks, properties = signal.find_peaks(y_list_shift, prominence=prom, width=(None, None))

    y = [y_list[i] for i in peaks]
    x = [x_list[i] for i in peaks]

    if test == 1:
        if full_only != 1:
            plt.figure(figsize=(12,8))
            plt.plot(x_list, y_list_shift)
            plt.title("shifted (check through)")
            plt.show()

    ranges = []
    for i,j in zip(properties["widths"], peaks):
        res_range = []
        if int(j-scale*i) > 0:
            res_range.append(int(j-scale*i))
        else:
            res_range.append(0)
        if int(j+scale*i) < len(wavelength_drop)-1:
            res_range.append(int(j+scale*i))
        else:
            res_range.append(len(wavelength_drop)-1)
        ranges.append(res_range)

    if test == 1:
        if full_only != 1:
            plt.figure(figsize=(12,8))
            plt.plot(x_list, y_list)
            plt.plot(x, y, "x")
            plt.hlines(y=[mean(y_list) for i in peaks], xmin=np.array([x_list[ranges[i][0]] for i in range(len(ranges))]), xmax=np.array([x_list[ranges[i][1]] for i in range(len(ranges))]))
            plt.show()

    wavelength_ranges = []
    for i in ranges:
        ran = []
        ran = [x_list[i[0]], x_list[i[1]]]
        wavelength_ranges.append(ran)

    return wavelength_ranges

def fitting_ranges(x_ranges, x_list, y_list):

    res = x_ranges
    res_wave = []
    res_loss = []
    q_wave = []

    for i in res:
        x_range = []
        y_range = []
        for j in x_list:
            if j >= i[0] and j <= i[1]:
                k = np.where(x_list==j)[0]
                x_range.append(x_list[k][0])
                y_range.append(y_list[k][0])
        q_wave.append(x_range)
        res_wave += x_range
        res_loss += y_range

    return([res_wave, res_loss, q_wave])

def shift_range(wave_t, wave_d, loss_t, loss_d):

    wave_t = list(wave_t)
    wave_t = [i+fitting_shift for i in wave_t]
    wave_t = [round(i, 4) for i in wave_t]
    wave_d = list(wave_d)
    wave_d = [round(i, 4) for i in wave_d]
    loss_t = list(loss_t)
    loss_d = list(loss_d)

    wave_range = [wave_t[0], wave_d[-1]]
    wave_t_range = [0, wave_t.index(wave_range[1])]
    wave_d_range = [wave_d.index(wave_range[0]), wave_d.index(wave_d[-1])]

    xt = wave_t[wave_t_range[0]: wave_t_range[1]]
    xd = wave_d[wave_d_range[0]: wave_d_range[1]]
    yt = loss_t[wave_t_range[0]: wave_t_range[1]]
    yd = loss_d[wave_d_range[0]: wave_d_range[1]]

    return(np.array(xt), np.array(xd), np.array(yt), np.array(yd), wave_t_range, wave_d_range)

def total_range (t_range, d_range, loss_through, loss_drop, wavelength_drop, wavelength_through):

    wavelength_through = list(wavelength_through)
    wavelength_drop = list(wavelength_drop)
    full = t_range + list(set(d_range) - set(t_range))

    ind = []
    for i in full:
        k = wavelength_through.index(i)
        ind.append(k)
    return [np.array(full), np.array(ind)]

def through(p, x):

    # print("through only")
    r = R # radius
    L_rt = 2*pi*r # round trip length

    x = np.array([i*10**-9 for i in x])
    phi = (2*pi/x)*p[5]*L_rt
    reduce_phi = phi%(2*pi)

    if p[0] < 0 or p[0] > 1:
        return np.array([1000 for i in x])

    alpha_wg = -log(10**(-p[0]/10))
    a = sqrt(math.exp(-alpha_wg*L_rt))
    A = math.exp(-alpha_wg*100*L_rt)

    k_1 = p[1]
    k_2 = p[1]

    if p[1] < 1 and p[1] > 0:
        t1 = sqrt(1-k_1**2)
        t2 = sqrt(1-k_2**2)

    else:
        return np.array([1000 for i in phi])

    Ethru = (t1-t2*sqrt(A)*np.exp(1j*(reduce_phi+p[2])))/(1-sqrt(A)*t1*t2*np.exp(1j*(reduce_phi+p[2])))
    Ethru_sq = Ethru*Ethru.conjugate()
    Tthru = p[3]*Ethru_sq.real+p[4]
    return Tthru

def through_background(p, x):

    # print("through with background")
    r = R # radius
    L_rt = 2*pi*r # round trip length

    x = np.array([i*10**-9 for i in x])
    phi = (2*pi/x)*p[5]*L_rt
    reduce_phi = phi%(2*pi)

    if p[0] < 0.1 or p[0] > 1:
        return np.array([1000 for i in x])

    alpha_wg = -log(10**(-p[0]/10))
    a = sqrt(math.exp(-alpha_wg*L_rt))
    A = math.exp(-alpha_wg*100*L_rt)

    k_1 = p[1]
    k_2 = p[1]

    if p[1] < 1 and p[1] > 0.1:
        t1 = sqrt(1-k_1**2)
        t2 = sqrt(1-k_2**2)

    else:
        return np.array([1000 for i in phi])

    Ethru = (t1-t2*sqrt(A)*np.exp(1j*(reduce_phi+p[2])))/(1-sqrt(A)*t1*t2*np.exp(1j*(reduce_phi+p[2])))
    Ethru_sq = Ethru*Ethru.conjugate()
    Thru_total = p[3]*Ethru_sq.real+p[4]+p[6]*(np.sin(p[7]*x-p[8]))+p[9]
    return Thru_total

def drop(p, x):

    r = R # radius
    L_rt = 2*pi*r # round trip length

    x = np.array([i*10**-9 for i in x])
    phi = (2*pi/x)*p[5]*L_rt
    reduce_phi = phi%(2*pi)

    if p[0] < 0.1 or p[0] > 1:
        return np.array([1000 for i in x])

    alpha_wg = -log(10**(-p[0]/10))
    a = sqrt(math.exp(-alpha_wg*L_rt))
    A = math.exp(-alpha_wg*100*L_rt)

    k_1 = p[1]
    k_2 = p[1]

    if p[1] < 1 and p[1] > 0.1:
        t1 = sqrt(1-k_1**2)
        t2 = sqrt(1-k_2**2)

    else:
        return np.array([1000 for i in phi])

    Edrop = -(k_1)*k_2*sqrt(sqrt(A))*np.exp(1j*(reduce_phi+p[2])/2)/(1-sqrt(A)*(t1)*(t2)*np.exp(1j*(reduce_phi+p[2])))
    Edrop_sq = Edrop*Edrop.conjugate()
    T_drop = p[3]*Edrop_sq.real+p[4]
    return (T_drop)

def sinusoidal(p, x):

    r = R # radius
    L_rt = 2*pi*r # round trip length

    x = np.array([i*10**-9 for i in x])
    phi = (2*pi/x)*p[5]*L_rt
    reduce_phi = phi%(2*pi)

    Sinu = p[6]*(np.sin(p[7]*x-p[8]))+p[9]
    return Sinu

def through_background_residual(params, x, data):

    alpha = params['alpha'].value
    kappa = params['kappa'].value
    through_x_offset = params['through_x_offset'].value
    through_y_scaling = params['through_y_scaling'].value
    through_y_offset = params['through_y_offset'].value
    effective_n = params['effective_n'].value
    sinusoidal_amplitude = params['sinusoidal_amplitude'].value
    sinusoidal_frequency = params['sinusoidal_frequency'].value
    sinusoidal_x_offset = params['sinusoidal_x_offset'].value
    sinusoidal_y_offset = params['sinusoidal_y_offset'].value

    p_t = [alpha, kappa, through_x_offset, through_y_scaling, through_y_offset, effective_n, sinusoidal_amplitude, sinusoidal_frequency, sinusoidal_x_offset, sinusoidal_y_offset]
    res_t_back = list(data - through_background(p_t, x))
    return res_t_back

def through_residual(params, x, data):

    alpha = params['alpha'].value
    kappa = params['kappa'].value
    through_x_offset = params['through_x_offset'].value
    through_y_scaling = params['through_y_scaling'].value
    through_y_offset = params['through_y_offset'].value
    effective_n = params['effective_n'].value

    p_t = [alpha, kappa, through_x_offset, through_y_scaling, through_y_offset, effective_n]
    res_t = list(data - through(p_t, x))
    return res_t

def drop_residual(params, x, data):

    alpha = params['alpha'].value
    kappa = params['kappa'].value
    drop_x_offset = params['drop_x_offset'].value
    drop_y_scaling = params['drop_y_scaling'].value
    drop_y_offset = params['drop_y_offset'].value
    effective_n = params['effective_n'].value

    p_d = [alpha, kappa, drop_x_offset, drop_y_scaling, drop_y_offset, effective_n]
    res_d = list(data - drop(p_d, x))

    return res_d

def total_residual(params, x, data):

    x_t = x[0]
    x_d = x[1]

    alpha = params['alpha'].value
    kappa = params['kappa'].value
    through_x_offset = params['through_x_offset'].value
    through_y_scaling = params['through_y_scaling'].value
    through_y_offset = params['through_y_offset'].value
    effective_n = params['effective_n'].value
    drop_x_offset = params['drop_x_offset'].value
    drop_y_scaling = params['drop_y_scaling'].value
    drop_y_offset = params['drop_y_offset'].value

    p_t = [alpha, kappa, through_x_offset, through_y_scaling, through_y_offset, effective_n]
    p_d = [alpha, kappa, drop_x_offset, drop_y_scaling, drop_y_offset, effective_n]
    res_t = list(data[0] - through(p_t, x_t))
    res_d = list(data[1] - drop(p_d, x_d))
    res = res_t + res_d

    return res

def transmission_fitting(type, radius, x_list, ranges, y_list, y_err, p_0, title, through_background_toggle):

    print(p_0)

    x = ranges[0]
    data = ranges[1]

    r = radius
    if test == 1:
        if full_only != 1:
            plt.figure(figsize=(15,10))
            plt.plot(x_list, y_list, marker=".")
            plt.plot(ranges[0], ranges[1])
            plt.title("General plot of data")
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Insertion Loss')
            plt.show()

    if type == "through":
        fit_params = Parameters()
        fit_params.add('alpha', value=p_0[0], min=p_0[0]-0.3*abs(p_0[0]), max=p_0[0]+0.3*abs(p_0[0]))
        fit_params.add('kappa', value=p_0[1], min=p_0[1]-0.25*abs(p_0[1]), max=p_0[1]+0.2*abs(p_0[1]))
        if p_0[2] == 0:
            fit_params.add('through_x_offset', value=p_0[2])
        else:
            fit_params.add('through_x_offset', value=p_0[2], min=p_0[2]-0.25*abs(p_0[2]), max=p_0[2]+0.25*abs(p_0[2]))
        fit_params.add('through_y_scaling', value=p_0[3], min=0)
        if p_0[4] == 0:
            fit_params.add('through_y_offset', value=p_0[4])
        else:
            fit_params.add('through_y_offset', value=p_0[4], min=p_0[4]-0.25*abs(p_0[4]), max=p_0[4]+0.25*abs(p_0[4]))
        fit_params.add('effective_n', value=p_0[5], min=3.64, max=3.66)

        if through_background_toggle == 1:
            fit_params.add('sinusoidal_amplitude', value=p_0[6])
            fit_params.add('sinusoidal_frequency', value=p_0[7])
            fit_params.add('sinusoidal_x_offset', value=p_0[8])
            fit_params.add('sinusoidal_y_offset', value=p_0[9])

            result = minimize(through_background_residual, fit_params, args=(x, data))
            report_fit(result.params)
            f.write(" \n")
            f.write("Through Background Fit: \n")
            for i in result.params:
                f.write(str(i) + ": " + str(result.params[i].value) + "\n")

            sinusoidal_amplitude = result.params['sinusoidal_amplitude'].value
            sinusoidal_frequency = result.params['sinusoidal_frequency'].value
            sinusoidal_x_offset = result.params['sinusoidal_x_offset'].value
            sinusoidal_y_offset = result.params['sinusoidal_y_offset'].value

        else:
            result = minimize(through_residual, fit_params, args=(x, data))
            report_fit(result.params)
            f.write(" \n")
            f.write("Through Background Fit: \n")
            for i in result.params:
                f.write(str(i) + ": " + str(result.params[i].value) + "\n")

        alpha = result.params['alpha'].value
        kappa = result.params['kappa'].value
        through_x_offset = result.params['through_x_offset'].value
        through_y_scaling = result.params['through_y_scaling'].value
        through_y_offset = result.params['through_y_offset'].value
        effective_n = result.params['effective_n'].value

        if through_background_toggle == 1:
            pf = [alpha, kappa, through_x_offset, through_y_scaling, through_y_offset, effective_n, sinusoidal_amplitude, sinusoidal_frequency, sinusoidal_x_offset, sinusoidal_y_offset]

        elif through_background_toggle == 0:
            pf = [alpha, kappa, through_x_offset, through_y_scaling, through_y_offset, effective_n]

    elif type == "drop":
        fit_params = Parameters()
        fit_params.add('alpha', value=p_0[0], min=p_0[0]-0.3*abs(p_0[0]), max=p_0[0]+0.3*abs(p_0[0]))
        fit_params.add('kappa', value=p_0[1], min=p_0[1]-0.25*abs(p_0[1]), max=p_0[1]+0.2*abs(p_0[1]))
        if p_0[2] == 0:
            fit_params.add('drop_x_offset', value=p_0[2])
        else:
            fit_params.add('drop_x_offset', value=p_0[2], min=p_0[2]-0.25*abs(p_0[2]), max=p_0[2]+0.25*abs(p_0[2]))
        fit_params.add('drop_y_scaling', value=p_0[3], min=0.5*p_0[3], max=2*p_0[3])
        if p_0[4] == 0:
            fit_params.add('drop_y_offset', value=p_0[4])
        else:
            fit_params.add('drop_y_offset', value=p_0[4], min=p_0[4]-0.25*abs(p_0[4]), max=p_0[4]+0.25*abs(p_0[4]))
        fit_params.add('effective_n', value=p_0[5], min=3.64, max=3.66)

        result = minimize(drop_residual, fit_params, args=(x, data))
        report_fit(result.params)
        f.write(" \n")
        f.write("Through Background Fit: \n")
        for i in result.params:
            f.write(str(i) + ": " + str(result.params[i].value) + "\n")

        alpha = result.params['alpha'].value
        kappa = result.params['kappa'].value
        drop_x_offset = result.params['drop_x_offset'].value
        drop_y_scaling = result.params['drop_y_scaling'].value
        drop_y_offset = result.params['drop_y_offset'].value
        effective_n = result.params['effective_n'].value

        pf = [alpha, kappa, drop_x_offset, drop_y_scaling, drop_y_offset, effective_n]

    else:
        return ("Enter for type either 'through' or 'drop'")

    if full_only != 1:
        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(111)
        ax.plot(x_list, y_list, marker=".", label="Data", Zorder=1, alpha=0.25)
        X = np.linspace(min(x_list), max(x_list), 1000)
        if type == "through":
            if through_background_toggle == 1:
                Y = through_background(pf, X)
            else:
                Y = through(pf, X)
        elif type == "drop":
            Y = drop(pf, X)
        ax.plot(X, Y, 'r-', label = 'Fit', Zorder=2)

        ax.set_title(title)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Insertion Loss')

        if through_background_toggle == 0:
            q_i = []
            q_c = []
            for i in ranges[2]:
                if type == "through":
                    y = list(through(pf, i))
                    peak_min = min(y)
                    index = y.index(peak_min)
                elif type == "drop":
                    y = list(drop(pf, i))
                    peak_max = max(y)
                    index = y.index(peak_max)
                lamb = i[index]*10**-7

                alpha = -log(10**(-pf[0]/10))
                Qi = 2*pi*pf[5]/lamb/alpha
                q_i.append(Qi)

                L = L_rt*10**2
                t1 = sqrt(1-pf[1]**2)
                Qc = -(pi*L*pf[5])/(lamb*log(abs(t1)))
                q_c.append(Qc)

            Qi = mean(q_i)
            Qc = mean(q_c)
            Q_total = 1/(1/Qc+1/Qi)

            print("Intrinsic Average: " + '%.3E' % Decimal(Qi))
            print("Coupling Average: " + '%.3E' % Decimal(Qc))
            print("Total: " + '%.3E' % Decimal(Q_total))

            f.write("Intrinsic Average: " + '%.3E\n' % Decimal(Qi))
            f.write("Coupling Average: " + '%.3E\n' % Decimal(Qc))
            f.write("Total: " + '%.3E\n' % Decimal(Q_total))

            Q_i = "Intrinsic: " + '%.3E' % Decimal(Qi)
            Q_c = "Coupling: " + '%.3E' % Decimal(Qc)
            Q_t = "Total: " + '%.3E' % Decimal(Q_total)

            if type == "through":
                plt.text(0.05, 0.2, Q_i, transform=ax.transAxes, fontsize=20, verticalalignment='top')
                if Q_all == 1:
                    plt.text(0.05, 0.15, Q_c, transform=ax.transAxes, fontsize=20, verticalalignment='top')
                    plt.text(0.05, 0.1, Q_t, transform=ax.transAxes, fontsize=20, verticalalignment='top')
                plt.legend(loc="lower right")
            if type == "drop":
                plt.text(0.05, 0.95, Q_i, transform=ax.transAxes, fontsize=20, verticalalignment='top')
                if Q_all == 1:
                    plt.text(0.05, 0.9, Q_c, transform=ax.transAxes, fontsize=20, verticalalignment='top')
                    plt.text(0.05, 0.85, Q_t, transform=ax.transAxes, fontsize=20, verticalalignment='top')
                plt.legend(loc="upper right")
        if save == 1:
            if type == "through":
                if through_background_toggle == 0:
                    plt.savefig(title + "_through_model.svg")
                elif through_background_toggle == 1:
                    plt.savefig(title + "_through_background_model.svg")
            elif type == "drop":
                plt.savefig(title + "_drop_model.svg")
        plt.show()
    return pf

def both_fit(radius, x_list, range_t, range_d, y_list_t, y_list_t_full, y_list_d, y_err, p_0, title):

    r = radius
    ran = shift_range(x_list, x_list, y_list_t, y_list_d)
    range_t = [i+fitting_shift for i in range_t]
    ranges_t = fitting_ranges(range_t, ran[0], ran[2])
    ranges_d = fitting_ranges(range_d, ran[1], ran[3])

    if test == 1:
        plt.figure(figsize=(15,10))
        plt.plot(x_list, y_list_t, marker=".")
        plt.plot(x_list, y_list_d, marker=".")
        plt.plot(ran[0], ran[2], marker=".")
        plt.plot(ran[1], ran[3], marker=".")
        plt.show()

    full = total_range(ranges_t[0], ranges_d[0], ran[2], ran[3], ran[1], ran[0])

    x_ranges = full[0]
    indices = full[1]
    y_ranges_t = [ran[2][i] for i in indices]
    y_ranges_d = [ran[3][i] for i in indices]

    if test == 1:
        plt.figure(figsize=(15,10))
        plt.plot(x_list, y_list_t, marker=".")
        plt.plot(x_list, y_list_d, marker=".")
        plt.plot(ran[0], ran[2], marker=".")
        plt.plot(ran[1], ran[3], marker=".")
        plt.scatter(ranges_t[0], ranges_t[1], marker="^", color="k", Zorder=5)
        plt.scatter(ranges_d[0], ranges_d[1], marker="^", color="k", Zorder=5)
        plt.show()

    fit_params = Parameters()
    fit_params.add('alpha', value=p_0[0], min=p_0[0]-0.25*abs(p_0[0]), max=p_0[0]+0.25*abs(p_0[0]))
    fit_params.add('kappa', value=p_0[1], min=p_0[1]-0.25*abs(p_0[1]), max=p_0[1]+0.25*abs(p_0[1]))
    if p_0[2] == 0:
        fit_params.add('through_x_offset', value=p_0[2])
    else:
        fit_params.add('through_x_offset', value=p_0[2], min=p_0[2]-0.25*abs(p_0[2]), max=p_0[2]+0.25*abs(p_0[2]))
    fit_params.add('through_y_scaling', value=p_0[3], min=0)
    if p_0[4] == 0:
        fit_params.add('through_y_offset', value=p_0[4])
    else:
        fit_params.add('through_y_offset', value=p_0[4], min=p_0[4]-0.25*abs(p_0[4]), max=p_0[4]+0.25*abs(p_0[4]))
    fit_params.add('effective_n', value=p_0[5], min=3.64, max=3.66)
    if p_0[6] == 0:
        fit_params.add('drop_x_offset', value=p_0[6])
    else:
        fit_params.add('drop_x_offset', value=p_0[6], min=p_0[6]-0.25*abs(p_0[6]), max=p_0[6]+0.25*abs(p_0[6]))
    fit_params.add('drop_y_scaling', value=p_0[7], min=0, max=p_0[7]*2)
    if p_0[8] == 0:
        fit_params.add('drop_y_offset', value=p_0[8])
    else:
        fit_params.add('drop_y_offset', value=p_0[8], min=p_0[8]-0.25*abs(p_0[8]), max=p_0[8]+0.25*abs(p_0[8]))

    x = [ranges_t[0], ranges_d[0]]
    data = [ranges_t[1], ranges_d[1]]

    result = minimize(total_residual, fit_params, args=(x, data))
    report_fit(result.params)
    f.write(" \n")
    f.write("Full Fit: \n")
    for i in result.params:
        f.write(str(i) + ": " + str(result.params[i].value) + "\n")

    alpha = result.params['alpha'].value
    kappa = result.params['kappa'].value
    through_x_offset = result.params['through_x_offset'].value
    through_y_scaling = result.params['through_y_scaling'].value
    through_y_offset = result.params['through_y_offset'].value
    effective_n = result.params['effective_n'].value
    drop_x_offset = result.params['drop_x_offset'].value
    drop_y_scaling = result.params['drop_y_scaling'].value
    drop_y_offset = result.params['drop_y_offset'].value

    p_t = [alpha, kappa, through_x_offset, through_y_scaling, through_y_offset, effective_n]
    p_d = [alpha, kappa, drop_x_offset, drop_y_scaling, drop_y_offset, effective_n]

    y_t = through(p_t, ran[0])
    y_d = drop(p_d, ran[1])

    if test == 1:
        plt.figure(figsize=(15,10))
        plt.plot(ran[0], ran[2], marker=".")
        plt.plot(ran[1], ran[3], marker=".")
        plt.plot(ran[0], y_t)
        plt.plot(ran[1], y_d)
        plt.show()

    x_t_shift = [i-fitting_shift for i in ran[0]]

    q_i = []
    q_ct = []
    q_cd = []

    for i in ranges_t[2]:
        yt = list(through(p_t, i))
        peak_min = min(yt)
        index_t = yt.index(peak_min)
        lamb_t = i[index_t]*10**-7

    for i in ranges_d[2]:
        yd = list(drop(p_d, i))
        peak_max = max(yd)
        index_d = yd.index(peak_max)
        lamb_d = i[index_d]*10**-7

        alpha = -log(10**(-p_t[0]/10))
        Qi_t = 2*pi*p_t[5]/lamb_t/alpha
        Qi_d = 2*pi*p_t[5]/lamb_d/alpha
        q_i.append(Qi_t)
        q_i.append(Qi_d)

        L = L_rt*10**2
        t1 = sqrt(1-p_t[1]**2)
        Qc_t = -(pi*L*p_t[5])/(lamb_t*log(abs(t1)))
        Qc_d = -(pi*L*p_t[5])/(lamb_d*log(abs(t1)))
        q_ct.append(Qc_t)
        q_cd.append(Qc_d)

    Qi = mean(q_i)
    Qct = mean(q_ct)
    Qcd = mean(q_cd)
    Qc = 1/(1/Qct+1/Qcd)
    Q_total = 1/(1/Qc+1/Qi)

    print(" ")
    print("Intrinsic Average: " + '%.3E' % Decimal(Qi))
    print("Coupling Through: " + '%.3E' % Decimal(Qct))
    print("Coupling Drop: " + '%.3E' % Decimal(Qcd))
    print("Coupling Total: " + '%.3E' % Decimal(Qc))
    print("Total: " + '%.3E' % Decimal(Q_total))

    f.write("\n")
    f.write("Intrinsic Average: " + '%.3E\n' % Decimal(Qi))
    f.write("Coupling Through: " + '%.3E\n' % Decimal(Qct))
    f.write("Coupling Drop: " + '%.3E\n' % Decimal(Qcd))
    f.write("Coupling Total: " + '%.3E\n' % Decimal(Qc))
    f.write("Total: " + '%.3E\n' % Decimal(Q_total))

    Q_i = "Intrinsic: " + '%.3E' % Decimal(Qi)
    Q_c = "Coupling: " + '%.3E' % Decimal(Qc)
    Q_t = "Total: " + '%.3E' % Decimal(Q_total)

    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    ax.plot(x_list, y_list_t_full, marker=".", color="C0", label="Raw Through Data", alpha=0.75, Zorder=1)
    ax.plot(x_list, y_list_t, marker=".", color="C1", label="Cleaned Through Data", alpha=0.75, Zorder=1)
    ax.plot(x_list, y_list_d, marker=".", color="C2", label="Drop Data", alpha=0.75, Zorder=1)
    ax.plot(x_t_shift, y_t, color="C3", label="Through Fit", Zorder=2)
    ax.plot(ran[1], y_d, color="C3", label="Drop Fit", Zorder=2)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Insertion Loss")
    ax.legend(loc="lower right")
    ax.set_title(title)
    props = dict(facecolor='white', alpha=0.9)
    if Q_all == 1:
        text = "{}\n{}\n{}".format(Q_i, Q_c, Q_t)
        plt.text(0.02, 0.15, text, transform=ax.transAxes, fontsize=20, verticalalignment='top', bbox=props)
    else:
        plt.text(0.02, 0.15, Q_i, transform=ax.transAxes, fontsize=20, verticalalignment='top', bbox=props)
    if save == 1:
        plt.savefig(title + "_full_model.svg")
    plt.show()

# Determing the wavelength ranges of interest for transmission_fitting
through_background_range = [[wavelength_through[0], wavelength_through[-1]]]
if manual_through_toggle == 1:
    through_background_range = through_manual
ranges_background_t = fitting_ranges(through_background_range, wavelength_through, loss_through)

print("Through Background: ")
f.write("Through Background: \n")
through_background_fit = transmission_fitting("through", R, wavelength_through, ranges_background_t, loss_through, dloss_through, p0_t, title, 1)
print(" ")
f.write("\n")

background = sinusoidal(through_background_fit, wavelength_through)
y_cleaned = np.array([i-j for i,j in zip(loss_through, background)])
y_cleaned_shift = np.array([shift_through+(10**(-i/10)) for i in y_cleaned])
through_range = resonance_ranges(wavelength_through, y_cleaned, y_cleaned_shift, prominence_through, scale_through)
if manual_through_toggle == 1:
    through_range = through_manual
ranges_t = fitting_ranges(through_range, wavelength_through, y_cleaned)

if test == 1:
    plt.figure(figsize = (15,8))
    plt.plot(wavelength_through, loss_through)
    plt.plot(wavelength_through, through(through_background_fit, wavelength_through))
    plt.plot(wavelength_through, y_cleaned)
    plt.title("cleaned data")
    plt.show()

print("Through Cleaned: ")
f.write("Through Cleaned: \n")
through_fit = transmission_fitting("through", R, wavelength_through, ranges_t, y_cleaned, dloss_through, through_background_fit, title, 0)
through_fit = through_background_fit[0:6]
print(" ")
f.write("\n")

drop_range = resonance_ranges(wavelength_drop, loss_drop, loss_drop, prominence_drop, scale_drop)
if manual_drop_toggle == 1:
    drop_range = drop_manual
ranges_d = fitting_ranges(drop_range, wavelength_drop, loss_drop)
print("Drop: ")
f.write("Drop: \n")
drop_fit = transmission_fitting("drop", R, wavelength_drop, ranges_d, loss_drop, dloss_drop, p0_d, title, 0)
print(" ")
f.write("\n ")

pf_avg = [mean([i,j]) for i,j in zip(through_fit, drop_fit)]
pf_total = [pf_avg[0], pf_avg[1], (through_fit[2]+fitting_shift), through_fit[3], through_fit[4], pf_avg[5], drop_fit[2], drop_fit[3], drop_fit[4], through_background_fit[6], through_background_fit[7], through_background_fit[8], through_background_fit[9]]
pf_total = [drop_fit[0], drop_fit[1], (through_fit[2]+fitting_shift), through_fit[3], through_fit[4], pf_avg[5], drop_fit[2], drop_fit[3], drop_fit[4], through_background_fit[6], through_background_fit[7], through_background_fit[8], through_background_fit[9]]
print("pf total: ", pf_total)
f.write("pf total: {}\n".format(pf_total))
total_fit = both_fit(R, wavelength_through, through_range, drop_range, y_cleaned, loss_through, loss_drop, dloss_through, pf_total, title)
