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

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ["Garamond"]
rcParams['font.size'] = '20'

# Loading proper files
wavelength_through, loss_through = np.loadtxt("C:\\Users\\HopeLee\\Documents\\Data\\Ring Resonators\\LSRL THERM2 W3\\Top_Right\\3E bottom\\Fine Scans\\3E_fine_1553_0_max_through.csv", delimiter=",", unpack=True, skiprows=19)
wavelength_drop, loss_drop = np.loadtxt("C:\\Users\\HopeLee\\Documents\\Data\\Ring Resonators\\LSRL THERM2 W3\\Top_Right\\3E bottom\\Fine Scans\\3E_fine_1553_0_max_drop.csv", delimiter=",", unpack=True, skiprows=19)

title = "3E Top Right Max"

dloss_through = 10**(-2)
dloss_drop = 10**(-2)

R = 4.8*10**-3
L_rt = 2*pi*R #round trip lengths

# Toggles (1=yes, 0=no)
test = 1
Q_all = 1
save = 1
full_only = 0
manual = 0

# Parameters for through peak finding
prominence_through = 0.00003 # filter to select correct peaks
scale_through = 0.8 # scaling for width of wavelength ranges
shift_through = 0.1 # vertical shift to change dips to peaks to use function

prominence_drop = 0.00001
scale_drop = 1

shift_constant = 0
fitting_shift = round((wavelength_through[2]-wavelength_through[1])*10**9*shift_constant, 4)

manual_through_ranges = [[1544.646, 1544.656], [1544.693, 1544.697], [1544.78, 1544.785]]

# Fitting parameters:
p0_t = [0.5, 0.47, 2.0, 0.0004, 0.00, 3.6500]
p0_d = [0.5, 0.47, 3.7, 0.00025, 0.00, 3.6500]

# p[0] = alpha --> coupling loss
# p[0] = alpha --> coupling loss
# p[1] = k
# p[2] = x axis offset, in units of radians, wavelength shift
# p[3] = y scaling (linear loss)
# p[4] = y offset
# p[5] = effective n

# Setting up lists of data values, round to 4 decimal places for ease of calculations
loss_through = np.array([10**(-i/10) for i in loss_through])
loss_through_shift = np.array([shift_through+(10**(-i/10)) for i in loss_through])  #shift dips to peaks so python function works, only needed for through
wavelength_through = [i*10**9 for i in wavelength_through]
wavelength_through = np.array([round(i, 4) for i in wavelength_through])

loss_drop = np.array([10**(-i/10) for i in loss_drop])
wavelength_drop = [i*10**9 for i in wavelength_drop]
wavelength_drop = np.array([round(i, 4) for i in wavelength_drop])

f= open(title+'.txt',"w+")

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
    Tthru = p[3]*Ethru_sq.real+p[4]

    return (Tthru)

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

def total(p, x):
    p_t = [p[0], p[1], p[2], p[3], p[4], p[8]]
    p_d = [p[0], p[1], p[5], p[6], p[7], p[8]]
    total = [i+j for i,j in zip(drop(p_d, x), through(p_t, x))]
    return(np.array(total))

def through_residual(p, x, y, err):
    return (through(p, x)-y)/err

def drop_residual(p, x, y, err):
    return (drop(p, x)-y)/err

def total_residual(p, x, y, err):
    return (total(p, x)-y)/err

def transmission_fitting(type, radius, x_list, ranges, y_list, y_err, p_0, title):

    r = radius
    if test == 1:
        if full_only != 1:
            plt.figure(figsize=(15,10))
            plt.plot(x_list, y_list, marker=".")
            plt.plot(ranges[0], ranges[1])
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Insertion Loss')
            plt.show()

    if type == "through":
        pf, cov, info, mesg, success = optimize.leastsq(through_residual, p_0, args=(ranges[0], ranges[1], y_err), full_output=1)
    elif type == "drop":
        pf, cov, info, mesg, success = optimize.leastsq(drop_residual, p_0, args=(ranges[0], ranges[1], y_err), full_output=1)
    else:
        return ("Enter for type either 'through' or 'drop'")

    if cov is None:
        print('Fit did not converge')
        print('Success code:', success)
        print(mesg)

        f.write('Fit did not converge\n')
        f.write('Success code: {}\n'.format(success))
        f.write("{}\n".format(mesg))
    else:
        print('Fit Converged')
        chisq = sum(info['fvec']*info['fvec'])
        dof = len(y_list)-len(pf)
        pferr = [np.sqrt(abs(cov[i,i])) for i in range(len(pf))]
        print('Converged with chi-squared', chisq)
        print('Number of degrees of freedom, dof =',dof)
        print('Reduced chi-squared:', chisq/dof)
        print('Inital guess values:')
        print('  p0 =', p_0)
        print('Best fit values:')
        print('  pf =', pf)
        print('Uncertainties in the best fit values:')
        print('  pferr =', pferr)
        print(" ")

        f.write('Fit Converged\n')
        f.write('Converged with chi-squared: {}\n'.format(chisq))
        f.write('Number of degrees of freedom, dof = {}\n'.format(dof))
        f.write('Reduced chi-squared: {}\n'.format(chisq/dof))
        f.write('Inital guess values:\n')
        f.write('  p0 = {}\n'.format(p_0))
        f.write('Best fit values:\n')
        f.write('  pf = {}\n'.format(pf))
        f.write('Uncertainties in the best fit values:\n')
        f.write('  pferr = {}\n'.format(pferr))
        f.write("\n ")

        if full_only != 1:
            fig = plt.figure(figsize=(15,10))
            ax = fig.add_subplot(111)
            ax.plot(x_list, y_list, marker=".", label="Data", Zorder=1, alpha=0.25)
            X = np.linspace(min(x_list), max(x_list), 1000)
            if type == "through":
                Y = through(pf, X)
            elif type == "drop":
                Y = drop(pf, X)
            ax.plot(X, Y, 'r-', label = 'Fit', Zorder=2)

            ax.set_title(title)
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Insertion Loss')

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
                    plt.savefig(title + "_through_model.svg")
                elif type == "drop":
                    plt.savefig(title + "_drop_model.svg")
            plt.show()
        return pf

def both_fit(radius, x_list, range_t, range_d, y_list_t, y_list_d, y_err, p_0, title):

    r = radius
    ran = shift_range(x_list, x_list, y_list_t, y_list_d)
    range_t = [i+fitting_shift for i in range_t]
    ranges_t = fitting_ranges(range_t, ran[0], ran[2])
    ranges_d = fitting_ranges(range_d, ran[1], ran[3])

    y_total = [i+j for i,j in zip(ran[2], ran[3])]

    if test == 1:
        plt.figure(figsize=(15,10))
        plt.plot(x_list, y_list_t, marker=".")
        plt.plot(x_list, y_list_d, marker=".")
        plt.plot(ran[0], ran[2], marker=".")
        plt.plot(ran[1], ran[3], marker=".")
        plt.show()

    if test == 1:
        plt.figure(figsize=(15,10))
        plt.plot(ran[0], y_total, marker=".")
        plt.plot(ran[0], ran[2], marker=".")
        plt.plot(ran[1], ran[3], marker=".")
        plt.plot(ranges_t[0], ranges_t[1], marker='.')
        plt.plot(ranges_d[0], ranges_d[1], marker='.')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Insertion Loss')
        plt.show()

    full = total_range(ranges_t[0], ranges_d[0], ran[2], ran[3], ran[1], ran[0])

    x_ranges = full[0]
    indices = full[1]
    y_ranges = [y_total[i] for i in indices]

    if test == 1:
        plt.figure(figsize=(15,10))
        plt.scatter(x_ranges, y_ranges, marker="o", color="r")
        plt.plot(ran[0], y_total, marker=".")
        plt.show()

    pf, cov, info, mesg, success = optimize.leastsq(total_residual, p_0, args=(x_ranges, y_ranges, y_err), full_output=1)

    if cov is None:
        print('Fit did not converge')
        print('Success code:', success)
        print(mesg)

        f.write('Fit did not converge\n')
        f.write('Success code: {}\n'.format(success))
        f.write("{}\n".format(mesg))
    else:
        print('Fit Converged')
        chisq = sum(info['fvec']*info['fvec'])
        dof = len(x_ranges)-len(pf)
        pferr = [np.sqrt(abs(cov[i,i])) for i in range(len(pf))]
        print('Converged with chi-squared', chisq)
        print('Number of degrees of freedom, dof =',dof)
        print('Reduced chi-squared:', chisq/dof)
        print('Inital guess values:')
        print('  p0 =', p_0)
        print('Best fit values:')
        pf_t = [pf[0], pf[1], pf[2], pf[3], pf[4], pf[8]]
        pf_d = [pf[0], pf[1], pf[5], pf[6], pf[7], pf[8]]
        print('  pf =', pf)
        print('  pf through =', pf_t)
        print('  pf drop =', pf_d)
        print('Uncertainties in the best fit values:')
        print('  pferr =', pferr)
        print()

        f.write('Fit Converged\n')
        f.write('Converged with chi-squared: {}\n'.format(chisq))
        f.write('Number of degrees of freedom, dof = {}\n'.format(dof))
        f.write('Reduced chi-squared: {}\n'.format(chisq/dof))
        f.write('Inital guess values:\n')
        f.write('  p0 = {}\n'.format(p_0))
        f.write('Best fit values:\n')
        f.write('  pf = {}\n'.format(pf))
        f.write('  pf through = {}\n'.format(pf_t))
        f.write('  pf drop = {}\n'.format(pf_d))
        f.write('Uncertainties in the best fit values:\n')
        f.write('  pferr = {}\n'.format(pferr))
        f.write("\n ")

        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(111)
        ax.errorbar(ran[0], y_total, yerr=y_err, marker=".", capsize=2, label="Data", Zorder=1, alpha=0.25)
        X = np.linspace(min(x_list), max(x_list), 1000)
        Y = total(pf, X)
        ax.plot(X, Y, 'r-', label = 'Fit', Zorder=2)
        plt.show()

        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(111)
        ax.errorbar(ran[0], ran[2], marker=".", capsize=2, label="Through Data", Zorder=1, alpha=0.25)
        ax.errorbar(ran[1], ran[3], marker=".", capsize=2, label="Drop Data", Zorder=1, alpha=0.25)
        Y_t = through(pf_t, X)
        Y_d = drop(pf_d, X)
        ax.plot(X, Y_t, 'r-', label = 'Through Fit', Zorder=2)
        ax.plot(X, Y_d, 'g-', label = 'Drop Fit', Zorder=2)
        ax.set_title(title)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Insertion Loss')
        ax.legend(loc="center right")

        q_i = []
        q_ct = []
        q_cd = []

        for i in ranges_t[2]:
            y_t = list(through(pf_t, i))
            peak_min = min(y_t)
            index_t = y_t.index(peak_min)
            lamb_t = i[index_t]*10**-7

        for i in ranges_d[2]:
            y_d = list(drop(pf_d, i))
            peak_max = max(y_d)
            index_d = y_d.index(peak_max)
            lamb_d = i[index_d]*10**-7

            alpha = -log(10**(-pf[0]/10))
            Qi_t = 2*pi*pf[8]/lamb_t/alpha
            Qi_d = 2*pi*pf[8]/lamb_d/alpha
            q_i.append(Qi_t)
            q_i.append(Qi_d)

            L = L_rt*10**2
            t1 = sqrt(1-pf[1]**2)
            Qc_t = -(pi*L*pf[8])/(lamb_t*log(abs(t1)))
            Qc_d = -(pi*L*pf[8])/(lamb_d*log(abs(t1)))
            q_ct.append(Qc_t)
            q_cd.append(Qc_d)

        Qi = mean(q_i)
        Qct = mean(q_ct)
        Qcd = mean(q_cd)
        Qc = 1/(1/Qct+1/Qcd)
        Q_total = 1/(1/Qc+1/Qi)

        print("Intrinsic Average: " + '%.3E' % Decimal(Qi))
        print("Coupling Through: " + '%.3E' % Decimal(Qct))
        print("Coupling Drop: " + '%.3E' % Decimal(Qcd))
        print("Coupling Total: " + '%.3E' % Decimal(Qc))
        print("Total: " + '%.3E' % Decimal(Q_total))

        f.write("Intrinsic Average: " + '%.3E\n' % Decimal(Qi))
        f.write("Coupling Through: " + '%.3E\n' % Decimal(Qct))
        f.write("Coupling Drop: " + '%.3E\n' % Decimal(Qcd))
        f.write("Coupling Total: " + '%.3E\n' % Decimal(Qc))
        f.write("Total: " + '%.3E\n' % Decimal(Q_total))

        Q_i = "Intrinsic: " + '%.3E' % Decimal(Qi)
        Q_c = "Coupling: " + '%.3E' % Decimal(Qc)
        Q_t = "Total: " + '%.3E' % Decimal(Q_total)

        plt.text(0.02, 0.25, Q_i, transform=ax.transAxes, fontsize=20, verticalalignment='top')
        if Q_all == 1:
            plt.text(0.02, 0.2, Q_c, transform=ax.transAxes, fontsize=20, verticalalignment='top')
            plt.text(0.02, 0.15, Q_t, transform=ax.transAxes, fontsize=20, verticalalignment='top')
        if save == 1:
            plt.savefig(title + "_full_no_shift_model.svg")
        plt.show()

        # now to correct the vertical and horizontal shifts
        pt_shift = [pf[0], pf[1], pf[2], pf[3], 0, pf[8]]
        pd_shift = [pf[0], pf[1], pf[5], pf[6], 0, pf[8]]
        pf_shift = [pf[0], pf[1], pf[2], pf[3], pf[4]-pf[7], pf[5], pf[6], 0, pf[8]]

        f.write(" ")
        f.write("Deliberate fitting shift: ")
        f.write("   Fitting Shift: {}".format(fitting_shift))
        f.write("   Shift Constant: {}".format(shift_constant))

        if test == 1:
            fig = plt.figure(figsize=(15,10))
            ax = fig.add_subplot(111)
            ax.errorbar(ran[0], y_total, marker=".", capsize=2, label="Data", Zorder=1, alpha=0.25)
            X = np.linspace(min(ran[0]), max(ran[0]), 1000)
            Y = total(pf_shift, X)
            ax.plot(X, Y, 'r-', label = 'Fit', Zorder=2)
            plt.show()


        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(111)
        ax.plot(ran[0], ran[2], marker=".", label="Through Data", Zorder=1, alpha=0.25)
        ax.plot(ran[1], ran[3], marker=".", label="Drop Data", Zorder=1, alpha=0.25)
        Y_t = through(pt_shift, X)
        Y_d = drop(pd_shift, X)
        ax.plot(X, Y_t, 'r-', label = 'Through Fit', Zorder=2)
        ax.plot(X, Y_d, 'g-', label = 'Drop Fit', Zorder=2)
        ax.set_title(title)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Insertion Loss')
        ax.legend(loc="center right")
        props = dict(facecolor='white', alpha=0.9)

        plt.text(0.02, 0.25, Q_i, transform=ax.transAxes, fontsize=20, verticalalignment='top', bbox=props)
        if Q_all == 1:
            plt.text(0.02, 0.2, Q_c, transform=ax.transAxes, fontsize=20, verticalalignment='top', bbox=props)
            plt.text(0.02, 0.15, Q_t, transform=ax.transAxes, fontsize=20, verticalalignment='top', bbox=props)

        if save == 1:
            plt.savefig(title + "_full_model.svg")
        plt.show()

# Determing the wavelength ranges of interest
through_range = resonance_ranges(wavelength_through, loss_through, loss_through_shift, prominence_through, scale_through)
# print(through_range)
if manual == 1:
    through_range = manual_through_ranges
drop_range = resonance_ranges(wavelength_drop, loss_drop, loss_drop, prominence_drop, scale_drop)

ranges_t = fitting_ranges(through_range, wavelength_through, loss_through)
ranges_d = fitting_ranges(drop_range, wavelength_drop, loss_drop)

print("Through: ")
f.write("Through: \n")
through_fit = transmission_fitting("through", R, wavelength_through, ranges_t, loss_through, dloss_through, p0_t, title)
print(" ")
f.write("\n")
print("Drop: ")
f.write("Drop: \n")
drop_fit = transmission_fitting("drop", R, wavelength_drop, ranges_d, loss_drop, dloss_drop, p0_d, title)
print(" ")
f.write("\n ")

pf_avg = [mean([i,j]) for i,j in zip(through_fit, drop_fit)]
pf_total = [pf_avg[0], pf_avg[1], (through_fit[2]+fitting_shift), through_fit[3], through_fit[4], drop_fit[2], drop_fit[3], drop_fit[4], pf_avg[5]]
print("pf total: ", pf_total)
f.write("pf total: {}\n".format(pf_total))

total_fit = both_fit(R, wavelength_through, through_range, drop_range, loss_through, loss_drop, dloss_through, pf_total, title)
