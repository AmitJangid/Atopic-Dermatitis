import numpy as np
import matplotlib.pyplot as plt
import random as ra, time
import random as ra
from scipy.integrate import odeint
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
####################################################################################################
#
#
#
#
# Order Th1, Th2, D, K Cells
Kx=1.0;    Ky=0.5;   Kz=2.0
p_q=0.01;  p_r=1.0;  p_p=1.0
p_v=0.2 ;  p_w=0.8;  p_u = 0.3
d=0.2;     n=2.0

Time = np.linspace(0, 120, 15000)
# IFN-gamma network 
def series(p_b, n, Kx, initial):
    trials = 1
    for k in range(trials):
        def ato(Var, t):
            x, y, z = Var
            f1 = p_q + (p_b/Kx) * ((x**n)/(1+(x**n))) * ((y**n)/(1+(y**n))) * ((d**n)/(1+(d**n))) - p_v * x
            f2 = p_r + (p_b/Ky) * (1/(1+(x**n))) * (1/(1+(y**n))) * (1/(1+(d**n))) - p_w * y
            f3 = p_p + (p_b/Kz) * (1/(1+(x**n))) * (1/(1+(y**n))) * (1/(1+(d**n))) - p_u * z
            return [f1, f2, f3]

        Var0 = initial #[ra.uniform(0.01,2), ra.uniform(2,4.0), ra.uniform(1.0,1.3)]

        sol = odeint(ato, Var0, Time)

    return sol

#
#
#
#
#
#







PATH = './script/'

# Tick Label size and Axis label size
SIZE = 17; FontX = 20; FontY = 20; 
LWseries = 0.75; LWbifurcation = 1.5; MS = 1.5; ticks_size=14.0
N, M = 9, 3

plt.figure(figsize=(3.8667*3.0,3.15*2.5))
plt.subplots_adjust(left=0.070, right=0.990, top=0.965, bottom=0.085, hspace = 1.0, wspace=0.265)







##########
plt.subplot2grid((N, M), (0,0), colspan=1, rowspan=4)

data_01 = np.genfromtxt(PATH+'/data_2p/ifn_n_b_bistable.dat'); 
data_02 = np.genfromtxt(PATH+'/data_2p/ifn_n_b_transition.dat'); 
data_02_dict = dict(data_02); data_02_compare = []
for key in data_01[:,0]: data_02_compare.append(data_02_dict[key])

#plt.plot(data_02[:,0], data_02[:,1], '--k', markersize=5.0);

plt.fill_between(data_01[:,0],data_01[:,1], data_01[:,2], color=colors['darkgrey'])
plt.fill_between(data_01[:,0],data_01[:,1], color=colors['mediumblue'])
plt.fill_between(data_02[:42,0],data_02[:42,1], color=colors['mediumblue'])
plt.fill_between(data_02[:42,0],data_02[:42,1], 70.0, color=colors['red'])

plt.fill_between(data_01[:,0],data_01[:,2], 70.0, where=data_01[:,2]<data_02_compare, color=colors['red'])
plt.fill_between(data_01[:,0],data_01[:,2], data_02_compare, where=data_01[:,2]<data_02_compare, color=colors['mediumblue'])
plt.fill_between(data_01[19:,0],data_01[19:,2], 70.0, color=colors['red'])

plt.title('A', loc='left', fontsize=SIZE, weight='bold')
plt.xticks(fontsize=ticks_size); plt.yticks([0,15,30,45,60], fontsize=ticks_size)
plt.xlabel(r'$n$', fontsize=FontX)
plt.ylabel(r'$\bar{b}_{_{IFN\gamma}}$', fontsize=FontY)
plt.axis([1.2,2.2,0,70])


plt.text(1.5, 45.0, 'Chronic', fontsize=15, color='w')
plt.text(1.8, 2.0, 'Acute', fontsize=15, color='w')
#plt.text(1.74, 8.5, 'Bistable (acute)', fontsize=8, color='k', rotation=15)
plt.text(1.83, 22.0, 'Bistable', fontsize=15, color='k', rotation=35)

plt.plot(1.4, 5.0, 'o', color='w', markersize=10.0)
plt.text(1.45, 3.8, r'A$_1$', color='w', fontsize=12)
plt.plot(1.4, 30.0, 'o', color='w', markersize=10.0)
plt.text(1.4, 24.0, r'A$_2$', color='w', fontsize=12)
#plt.plot(2.15, 17.0, 'o', color='w', markersize=10.0)
#plt.text(2.05, 14.3, r'A$_3$', color='w', fontsize=12)
plt.plot(2.15, 30.0, 'o', color='w', markersize=10.0)
plt.text(2.10, 23.0, r'A$_3$', color='w', fontsize=12)





##########
plt.subplot2grid((N, M), (5,0), colspan=1, rowspan=4)

data_11 = np.genfromtxt(PATH+'/data_2p/ifn_kT1_b_bistable.dat'); 
data_12 = np.genfromtxt(PATH+'/data_2p/ifn_kT1_b_transition.dat'); 

#plt.plot(data_12[:,0], data_12[:,1], '--k', markersize=5.0);
plt.fill_between(data_11[:,0],data_11[:,1], color=colors['mediumblue'])
plt.fill_between(data_11[:,0],data_11[:,2], 70.0, color=colors['red'])
plt.fill_between(data_11[:,0],data_11[:,1], data_11[:,2], color=colors['darkgrey'])
plt.fill_between(data_12[:2,0],data_12[:2,1], 70.0, color=colors['red'])

plt.title('B', loc='left', fontsize=SIZE, weight='bold')
plt.xticks(fontsize=ticks_size); plt.yticks([0,15,30,45,60], fontsize=ticks_size)
plt.xlabel(r'$k_{T_1}$', fontsize=FontX)
plt.ylabel(r'$\bar{b}_{_{IFN\gamma}}$', fontsize=FontY)
plt.axis([0.01,2.0,0,70])

plt.text(0.25, 45.0, 'Chronic', fontsize=15, color='w')
plt.text(1.2, 5.0, 'Acute', fontsize=15, color='w')
#plt.text(1.05, 14.0, 'Bistable (acute)', fontsize=10, color='k', rotation=17)
plt.text(1.0, 25.0, 'Bistable', fontsize=15, color='k', rotation=27)

plt.plot(1.0, 5.0, 'o', color='w', markersize=10.0)
plt.text(0.8, 3.7, r'B$_1$', color='w', fontsize=12)
plt.plot(1.0, 50.0, 'o', color='w', markersize=10.0)
plt.text(1.12, 48.0, r'B$_2$', color='w', fontsize=12)
#plt.plot(1.9, 29.0, 'o', color='w', markersize=10.0)
#plt.text(1.82, 23.0, r'B$_3$', color='w', fontsize=12)
plt.plot(1.75, 42.0, 'o', color='w', markersize=10.0)
plt.text(1.75, 36.0, r'B$_3$', color='w', fontsize=12)


# fir bifurcation (Kx=1.50)
# sec bifurcation (Kx=0.25)















##########
ax2 = plt.subplot2grid((N, M), (0,1), colspan=1, rowspan=2)
'''
for trials in range(2):                                       # Simulation on fixed value of parameters set
    if trials ==0:
        initial = [ra.uniform(0.01,0.05), ra.uniform(3.5,4.0), ra.uniform(1.0,1.3)]
        data = series(5.0, 1.4, 1.0, initial)
        np.savetxt('./script/data_2p/series_B1_0.dat', data, fmt='%0.5f')  
        plt.plot(Time, data[:,0], '-r', Time, data[:,1], '-b', linewidth = LWseries)
    else: 
        initial = [ra.uniform(3.5,4.0), ra.uniform(0.0,0.05), ra.uniform(1.0,1.3)]
        data = series(5.0, 1.4, 1.0, initial)
        np.savetxt('./script/data_2p/series_B1_1.dat', data, fmt='%0.5f')  
        plt.plot(Time, data[:,0], '--r', Time, data[:,1], '--b', linewidth = LWseries)
'''
data = np.genfromtxt('./script/data_2p/series_B1_0.dat')
plt.plot(Time, data[:,0], '-r', Time, data[:,1], '-b', linewidth = LWseries)
data = np.genfromtxt('./script/data_2p/series_B1_1.dat')
plt.plot(Time, data[:,0], '--r', Time, data[:,1], '--b', linewidth = LWseries)
     
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False); plt.xlim([0,120])
plt.xticks([]); #plt.yticks([])
plt.title(r'A$_1$', loc='left', fontsize=SIZE-3, weight='bold')
plt.title(r'Acute', loc='right', fontsize=SIZE-3)
#label = ax2.set_ylabel(r'$\bar{P}_{_{T_2}}$', labelpad=0.0, fontsize=FontY-3, color='blue')
#ax2.yaxis.set_label_coords(-0.11, 0.05)
plt.ylabel('Population', fontsize=12, labelpad=15)

##########
ax3 = plt.subplot2grid((N, M), (2,1), colspan=1, rowspan=2)
'''
for trials in range(2):                                       # Simulation on fixed value of parameters set
    if trials ==0:
        initial = [ra.uniform(0.01,0.05), ra.uniform(14,15.0), ra.uniform(1.0,1.3)]
        data = series(30.0, 1.4, 1.0, initial)
        np.savetxt('./script/data_2p/series_B2_0.dat', data, fmt='%0.5f')  
        plt.plot(Time, data[:,0], '-r', Time, data[:,1], '-b', linewidth = LWseries)
    else: 
        initial = [ra.uniform(14,15.0), ra.uniform(0,0.05), ra.uniform(1.0,1.3)]
        data = series(30.0, 1.4, 1.0, initial)
        np.savetxt('./script/data_2p/series_B2_1.dat', data, fmt='%0.5f')  
        plt.plot(Time, data[:,0], '--r', Time, data[:,1], '--b', linewidth = LWseries)
'''
data = np.genfromtxt('./script/data_2p/series_B2_0.dat')
plt.plot(Time, data[:,0], '-r', Time, data[:,1], '-b', linewidth = LWseries)
data = np.genfromtxt('./script/data_2p/series_B2_1.dat')
plt.plot(Time, data[:,0], '--r', Time, data[:,1], '--b', linewidth = LWseries)

ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
plt.xlim([0,40])
#label = ax3.set_ylabel(r'$\bar{P}_{_{T_1}}$', labelpad=0.0, fontsize=FontY-3, color='red')
#ax3.yaxis.set_label_coords(-0.11, 0.95)
plt.title(r'A$_2$', loc='left', fontsize=SIZE-3, weight='bold')
plt.title(r'Chronic', loc='right', fontsize=SIZE-3)
plt.xlabel(r'Time', fontsize=FontX-2)
plt.ylabel('Population', fontsize=12, labelpad=9)

##########
'''
ax4 = plt.subplot2grid((N, M), (0,2), colspan=1, rowspan=2)
for trials in range(2):                                       # Simulation on fixed value of parameters set
    if trials ==0:
        initial = [ra.uniform(0.25,0.5), ra.uniform(5,6.0), ra.uniform(1.0,1.3)]
        data = series(17.0, 2.15, 1.0, initial)
        np.savetxt('./script/data_2p/series_B3_0.dat', data, fmt='%0.5f')  
        plt.plot(Time, data[:,0], '-r', Time, data[:,1], '-b', linewidth = LWseries)
    else: 
        initial = [ra.uniform(5,6.0), ra.uniform(0,0.05), ra.uniform(1.0,1.3)]
        data = series(17.0, 2.15, 1.0, initial)
        np.savetxt('./script/data_2p/series_B3_1.dat', data, fmt='%0.5f')  
        plt.plot(Time, data[:,0], '--r', Time, data[:,1], '--b', linewidth = LWseries)

data = np.genfromtxt('./script/data_2p/series_B3_0.dat')
plt.plot(Time, data[:,0], '-r', Time, data[:,1], '-b', linewidth = LWseries)
data = np.genfromtxt('./script/data_2p/series_B3_1.dat')
plt.plot(Time, data[:,0], '--r', Time, data[:,1], '--b', linewidth = LWseries)

label = ax4.set_ylabel(r'$\bar{P}_{_{T_2}}$', labelpad=0.0, fontsize=FontY-3, color='blue')
ax4.yaxis.set_label_coords(-0.11, 0.05)
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
plt.xticks([]); plt.xlim([0,60])
plt.title(r'A$_3$', loc='left', fontsize=SIZE-3, weight='bold')
plt.title(r'Bistable (Acute)', loc='right', fontsize=SIZE-3)
'''
##########
ax5 = plt.subplot2grid((N, M), (0,2), colspan=1, rowspan=2)
'''
for trials in range(2):                                       # Simulation on fixed value of parameters set
    if trials ==0:
        initial = [ra.uniform(0,0.5), ra.uniform(5,5.5), 1.2229178916871202]
        print (initial)
        data = series(30.0, 2.15, 1.0, initial)
        np.savetxt('./script/data_2p/series_B4_0.dat', data, fmt='%0.5f')  
        plt.plot(Time, data[:,0], '-r', Time, data[:,1], '-b', linewidth = LWseries)
    else: 
        initial = [0.25382325939173084, 3.1994365185815674, 1.2897195036841347]
        print (initial)
        data = series(30.0, 2.15, 1.0, initial)
        np.savetxt('./script/data_2p/series_B4_1.dat', data, fmt='%0.5f')  
        plt.plot(Time, data[:,0], '--r', Time, data[:,1], '--b', linewidth = LWseries)
'''
data = np.genfromtxt('./script/data_2p/series_B4_0.dat')
plt.plot(Time, data[:,0], '-r', Time, data[:,1], '-b', linewidth = LWseries)
data = np.genfromtxt('./script/data_2p/series_B4_1.dat')
plt.plot(Time, data[:,0], '--r', Time, data[:,1], '--b', linewidth = LWseries)

ax5.spines['right'].set_visible(False)
ax5.spines['top'].set_visible(False)
plt.xlabel(r'Time', fontsize=FontX-2)
#plt.xticks(fontsize=ticks_size)
plt.xlim([0,60])
#label = ax5.set_ylabel(r'$\bar{P}_{_{T_1}}$', labelpad=0.0, fontsize=FontY-3, color='red')
#ax5.yaxis.set_label_coords(-0.11, 0.95)
plt.title(r'A$_3$', loc='left', fontsize=SIZE-3, weight='bold')
plt.title(r'Bistable', loc='right', fontsize=SIZE-3)
plt.ylabel('Population', fontsize=12, labelpad=15)
plt.legend([r'$\bar{P}_{_{T_1}}$', r'$\bar{P}_{_{T_2}}$'], frameon=True, bbox_to_anchor=(0.30,-0.5), fontsize=12)

'''
##########
ax6 = plt.subplot2grid((N, M), (0,2), colspan=1, rowspan=2)
data = np.genfromtxt(PATH+'/data_2p/ifn_b_n_1.75.dat');

USb1 = []; Sb1L = []; Sb1U = []
for data in data:
    if data[1] > 0.121826 and data[1] < 0.7: USb1.append(data)
    elif data[1] <= 0.121826: Sb1L.append(data)
    else: Sb1U.append(data)

plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,1], '--',color='red', linewidth = LWbifurcation, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,2], '--',color='blue', linewidth = LWbifurcation, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,1], '-',color='red', linewidth = LWbifurcation, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,2], '-',color='blue', linewidth = LWbifurcation, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,1], '-',color='red', label=r'$\bar{P}_{T_1}$', linewidth = LWbifurcation, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,2], '-',color='blue', label=r'$\bar{P}_{T_2}$', linewidth = LWbifurcation, markersize=MS)

plt.title('C', loc='left', fontsize=SIZE, weight='bold')
ax6.spines['right'].set_visible(False)
ax6.spines['top'].set_visible(False)
plt.xticks([0,5,10,15,20,25], fontsize=7.6, weight='bold')
plt.text(0.0,4.0,r'$n$=1.75', fontsize=12)

##########
ax7 = plt.subplot2grid((N, M), (2,2), colspan=1, rowspan=2)
data = np.genfromtxt(PATH+'/data_2p/ifn_b_n_2.1.dat');

USb1 = []; Sb1L = []; Sb1U = []
for data in data:
    if data[1] > 0.0960 and data[1] < 0.945: USb1.append(data)
    elif data[1] <= 0.0960: Sb1L.append(data)
    else: Sb1U.append(data)

plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,1], '--',color='red', linewidth = LWbifurcation, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,2], '--',color='blue', linewidth = LWbifurcation, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,1], '-',color='red', linewidth = LWbifurcation, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,2], '-',color='blue', linewidth = LWbifurcation, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,1], '-',color='red', label=r'$\bar{P}_{T_1}$', linewidth = LWbifurcation, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,2], '-',color='blue', label=r'$\bar{P}_{T_2}$', linewidth = LWbifurcation, markersize=MS)

plt.xlabel(r'$b_{_{IFN\gamma}}$', fontsize=FontX)
label = ax7.set_ylabel(r'Steady state', labelpad=0.0, fontsize=FontY-7)
ax7.yaxis.set_label_coords(-0.10, 1.075)
plt.xticks([0, 15, 30, 45, 60, 75], fontsize=ticks_size)
plt.yticks([0, 2, 4, 6, 8])
plt.xlim([-5,75]); 
ax7.spines['right'].set_visible(False)
ax7.spines['top'].set_visible(False)
plt.text(0.0,6.0,r'$n$=2.1', fontsize=12)
'''




















##########
ax9 = plt.subplot2grid((N, M), (5,1), colspan=1, rowspan=2)
'''
for trials in range(2):                                       # Simulation on fixed value of parameters set
    if trials ==0:
        initial = [2.2, 0.8, 1.2229178916871202]
        print (initial)
        data = series(5.0, 2.0, 1.0, initial)
        np.savetxt('./script/data_2p/series_C1_0.dat', data, fmt='%0.5f')  
        plt.plot(Time, data[:,0], '-r', Time, data[:,1], '-b', linewidth = LWseries)
    else: 
        initial = [1, 4, 1.2897195036841347]
        print (initial)
        data = series(5.0, 2.0, 1.0, initial)
        np.savetxt('./script/data_2p/series_C1_1.dat', data, fmt='%0.5f')  
        plt.plot(Time, data[:,0], '--r', Time, data[:,1], '--b', linewidth = LWseries)
'''
data = np.genfromtxt('./script/data_2p/series_C1_0.dat')
plt.plot(Time, data[:,0], '-r', Time, data[:,1], '-b', linewidth = LWseries)
data = np.genfromtxt('./script/data_2p/series_C1_1.dat')
plt.plot(Time, data[:,0], '--r', Time, data[:,1], '--b', linewidth = LWseries)

ax9.spines['right'].set_visible(False)
ax9.spines['top'].set_visible(False)
plt.xlim([0,60]); plt.xticks([])
plt.title(r'B$_1$', loc='left', fontsize=SIZE-3, weight='bold')
plt.title(r'Acute', loc='right', fontsize=SIZE-3)
#label = ax9.set_ylabel(r'$\bar{P}_{_{T_2}}$', labelpad=0.0, fontsize=FontY-3, color='blue')
#ax9.yaxis.set_label_coords(-0.11, 0.05)
plt.ylabel('Population', fontsize=12, labelpad=15)

##########
ax10 = plt.subplot2grid((N, M), (7,1), colspan=1, rowspan=2)
'''
for trials in range(2):                                       # Simulation on fixed value of parameters set
    if trials ==0:
        initial = [2.2, 0.8, 1.2229178916871202]
        print (initial)
        data = series(50.0, 2.0, 1.0, initial)
        np.savetxt('./script/data_2p/series_C2_0.dat', data, fmt='%0.5f')  
        plt.plot(Time, data[:,0], '-r', Time, data[:,1], '-b', linewidth = LWseries)
    else: 
        initial = [10, 8, 1.2897195036841347]
        print (initial)
        data = series(50.0, 2.0, 1.0, initial)
        np.savetxt('./script/data_2p/series_C2_1.dat', data, fmt='%0.5f')  
        plt.plot(Time, data[:,0], '--r', Time, data[:,1], '--b', linewidth = LWseries)
'''
data = np.genfromtxt('./script/data_2p/series_C2_0.dat')
plt.plot(Time, data[:,0], '-r', Time, data[:,1], '-b', linewidth = LWseries)
data = np.genfromtxt('./script/data_2p/series_C2_1.dat')
plt.plot(Time, data[:,0], '--r', Time, data[:,1], '--b', linewidth = LWseries)

#label = ax10.set_ylabel(r'$\bar{P}_{_{T_2}}$', labelpad=0.0, fontsize=FontY-7, color='blue')
#ax10.yaxis.set_label_coords(-0.07, 0.25)
ax10.spines['right'].set_visible(False)
ax10.spines['top'].set_visible(False)
plt.xlim([0,60])
plt.title(r'B$_2$', loc='left', fontsize=SIZE-3, weight='bold')
plt.title(r'Chronic', loc='right', fontsize=SIZE-3)
#label = ax10.set_ylabel(r'$\bar{P}_{_{T_1}}$', labelpad=0.0, fontsize=FontY-3, color='red')
#ax10.yaxis.set_label_coords(-0.11, 0.95)
plt.xlabel(r'Time', fontsize=FontX-2)
plt.ylabel('Population', fontsize=12, labelpad=9)

##########
'''
ax11 = plt.subplot2grid((N, M), (5,2), colspan=1, rowspan=2)
for trials in range(2):                                       # Simulation on fixed value of parameters set
    if trials ==0:
        initial = [4.0580742497984295, 0.6956534448978327, 1.2850206954754864]
        print (initial)
        data = series(29.0, 2.0, 1.9, initial)
        np.savetxt('./script/data_2p/series_C3_0.dat', data, fmt='%0.5f')  
        plt.plot(Time, data[:,0], '-r', Time, data[:,1], '-b', linewidth = LWseries)
    else: 
        initial = [0.30481382824901404, 3.8723316635949097, 1.2363609756724412]
        print (initial)
        data = series(29.0, 2.0, 1.9, initial)
        np.savetxt('./script/data_2p/series_C3_1.dat', data, fmt='%0.5f')  
        plt.plot(Time, data[:,0], '--r', Time, data[:,1], '--b', linewidth = LWseries)
data = np.genfromtxt('./script/data_2p/series_C3_0.dat')
plt.plot(Time, data[:,0], '-r', Time, data[:,1], '-b', linewidth = LWseries)
data = np.genfromtxt('./script/data_2p/series_C3_1.dat')
plt.plot(Time, data[:,0], '--r', Time, data[:,1], '--b', linewidth = LWseries)

label = ax11.set_ylabel(r'$\bar{P}_{_{T_2}}$', labelpad=0.0, fontsize=FontY-3, color='blue')
ax11.yaxis.set_label_coords(-0.11, 0.05)
ax11.spines['right'].set_visible(False)
ax11.spines['top'].set_visible(False)
plt.xticks([]); plt.xlim([0,50])
plt.title(r'B$_3$', loc='left', fontsize=SIZE-3, weight='bold')
plt.title(r'Bistable (Acute)', loc='right', fontsize=SIZE-3)
'''
##########
ax12 = plt.subplot2grid((N, M), (5,2), colspan=1, rowspan=2)

for trials in range(2):                                       # Simulation on fixed value of parameters set
    if trials ==0:
        initial = [0.17427860637995196, 4.618653214721293, 1.8135213831075898]
        print (initial)
        data = series(42.0, 2.0, 1.75, initial)
        np.savetxt('./script/data_2p/series_C4_0.dat', data, fmt='%0.5f')  
        plt.plot(Time, data[:,0], '-r', Time, data[:,1], '-b', linewidth = LWseries)
    else: 
        initial = [0.15, 8, 1.720408488476664]
        print (initial)
        data = series(42.0, 2.0, 1.75, initial)
        np.savetxt('./script/data_2p/series_C4_1.dat', data, fmt='%0.5f')  
        plt.plot(Time, data[:,0], '--r', Time, data[:,1], '--b', linewidth = LWseries)

#data = np.genfromtxt('./script/data_2p/series_C4_0.dat')
#plt.plot(Time, data[:,0], '-r', Time, data[:,1], '-b', linewidth = LWseries)
#data = np.genfromtxt('./script/data_2p/series_C4_1.dat')
#plt.plot(Time, data[:,0], '--r', Time, data[:,1], '--b', linewidth = LWseries)

ax12.spines['right'].set_visible(False)
ax12.spines['top'].set_visible(False)
plt.xlabel(r'Time', fontsize=FontX-2)
#plt.xticks(fontsize=ticks_size)
plt.xlim([0,80])
#label = ax12.set_ylabel(r'$\bar{P}_{_{T_1}}$', labelpad=0.0, fontsize=FontY-3, color='red')
#ax12.yaxis.set_label_coords(-0.11, 0.95)
plt.title(r'B$_3$', loc='left', fontsize=SIZE-3, weight='bold')
plt.title(r'Bistable', loc='right', fontsize=SIZE-3)
plt.ylabel('Population', fontsize=12, labelpad=9)
plt.legend([r'$\bar{P}_{_{T_1}}$', r'$\bar{P}_{_{T_2}}$'], frameon=True, bbox_to_anchor=(0.30,-0.5), fontsize=12)

'''
##########
ax13 = plt.subplot2grid((N, M), (5,2), colspan=1, rowspan=2)
data = np.genfromtxt(PATH+'/data_2p/ifn_b_Kx_0.25.dat');

#plt.plot(data[:,0], data[:,1], '.r', data[:,0], data[:,2], '.b')

USb1 = []; Sb1L = []; Sb1U = []
for data in data:
    if data[1] > 0.1025 and data[1] < 0.8: USb1.append(data)
    elif data[1] <= 0.1025: Sb1L.append(data)
    else: Sb1U.append(data)

plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,1], '--',color='red', linewidth = LWbifurcation, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,2], '--',color='blue', linewidth = LWbifurcation, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,1], '-',color='red', linewidth = LWbifurcation, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,2], '-',color='blue', linewidth = LWbifurcation, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,1], '-',color='red', label=r'$\bar{P}_{T_1}$', linewidth = LWbifurcation, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,2], '-',color='blue', label=r'$\bar{P}_{T_2}$', linewidth = LWbifurcation, markersize=MS)

plt.title('F', loc='left', fontsize=SIZE, weight='bold')
ax13.spines['right'].set_visible(False)
ax13.spines['top'].set_visible(False)
plt.xticks([0,3,6,9,12,15], fontsize=7.6, weight='bold')
plt.yticks([0, 2, 4, 6, 8])
plt.text(0.0,6.0,r'$k_{T_{1}}$=0.25', fontsize=12)


##########
ax14 = plt.subplot2grid((N, M), (7,2), colspan=1, rowspan=2)
data = np.genfromtxt(PATH+'/data_2p/ifn_b_Kx_1.5.dat');

#plt.plot(data[:,0], data[:,1], '.r', data[:,0], data[:,2], '.b')

USb1 = []; Sb1L = []; Sb1U = []
for data in data:
    if data[1] > 0.103 and data[1] < 0.9: USb1.append(data)
    elif data[1] <= 0.103: Sb1L.append(data)
    else: Sb1U.append(data)

plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,1], '--',color='red', linewidth = LWbifurcation, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,2], '--',color='blue', linewidth = LWbifurcation, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,1], '-',color='red', linewidth = LWbifurcation, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,2], '-',color='blue', linewidth = LWbifurcation, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,1], '-',color='red', label=r'$\bar{P}_{T_1}$', linewidth = LWbifurcation, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,2], '-',color='blue', label=r'$\bar{P}_{T_2}$', linewidth = LWbifurcation, markersize=MS)

plt.xlabel(r'$b_{_{IFN\gamma}}$', fontsize=FontX)
label = ax14.set_ylabel(r'Steady state', labelpad=0.0, fontsize=FontY-7)
ax14.yaxis.set_label_coords(-0.10, 1.075)
plt.xticks([0, 10, 20, 30, 40, 50], fontsize=ticks_size)
ax14.spines['right'].set_visible(False)
ax14.spines['top'].set_visible(False)
plt.text(.0,4.0,r'$k_{T_{1}}$=1.5', fontsize=12)
'''









#plt.tight_layout()
plt.savefig('Figure4.pdf')
#plt.show()

####################################################################################################

