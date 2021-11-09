import numpy as np, random as ra
import matplotlib.pyplot as plt
from scipy.integrate import odeint
####################################################################################################
#
#
#
#
# Order Th1, Th2, D, K Cells
Kx=1.0;    Ky=0.5;   Kz=2.0
p_q=0.01;  p_r=1.0;  p_p=1.0
p_v=0.2 ;  p_w=0.8;  p_u = 0.3
d=0.2;     #n=2.0

input_drug = 5.0

Time = np.linspace(0, 105, 15000)
def series(n, p_b, initial):
    if n == 1.4 and p_b == 5.0: 
        Time = np.linspace(0, 205, 15000)
        StartIn, EndIn = 45, 65
        
    elif  n == 1.4 and p_b == 40.0:
        Time = np.linspace(0, 105, 15000)
        StartIn, EndIn = 45, 65
    
    else: 
        Time = np.linspace(0, 105, 15000)
        StartIn, EndIn = 45, 65
    
    
    def ato(Var, t):
        x, y, z = Var
        S = 0 if t < StartIn else input_drug if t >= StartIn and t <= EndIn else 0
        f1 = p_q +  (p_b/Kx) * ((x**n)/(1+(x**n))) * ((y**n)/(1+(y**n))) * ((d**n)/(1+(d**n)) * ((1.0)/(1+(S**n))) ) - p_v * x
        f2 = p_r +  (p_b/Ky) * (1/(1+(x**n))) * (1/(1+(y**n))) * (1/(1+(d**n))) - p_w * y
        f3 = p_p +  (p_b/Kz) * (1/(1+(x**n))) * (1/(1+(y**n))) * (1/(1+(d**n))) - p_u * z
        return [f1, f2, f3]


    Var0 = initial #[ra.uniform(2,3), ra.uniform(0,2.0), ra.uniform(1.0,1.3)]
 
    sol = odeint(ato, Var0, Time)

    return sol

#

#
#
#
#
#








# Tick Label size and Axis label size
SIZE = 17; FontX = 20; FontY = 20;
LWseries = 0.75; LWbifurcation = 1.5; MS = 1.5; ticks_size=14.0
N, M = 2, 2

plt.figure(figsize=(3.8667*3.0,3.15*2.1))
plt.subplots_adjust(left=0.045, right=0.997, top=0.955, bottom=0.090, hspace = 0.400, wspace=0.15)

ax1 = plt.subplot(N, M, 1)
S = [0 if t < 45 else input_drug if t >= 45 and t <= 65 else 0 for t in Time]
plt.plot(Time, S, '-k')
#plt.ylabel(r'Strength', fontsize=FontY-7)
plt.yticks([0.0, 5.0], fontsize=ticks_size)
plt.xticks([0, 25, 50, 75, 100], fontsize=ticks_size)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
plt.title('A', loc='left', fontsize=SIZE, weight='bold')
plt.ylabel(r'Strength', fontsize=FontY-4)


ax1 = plt.subplot(N, M, 3)
S = [0 if t < 45 else input_drug if t >= 45 and t <= 65 else 0 for t in Time]
plt.plot(Time, S, '-k')
#label = ax1.set_ylabel(r'Steady state', labelpad=0.0, fontsize=FontY-7)
#ax1.yaxis.set_label_coords(-0.10, 1.075)
plt.ylabel(r'Strength', fontsize=FontY-4)
plt.yticks([0.0, 5.0], fontsize=ticks_size)
plt.xticks([0, 25, 50, 75, 100], fontsize=ticks_size)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
plt.title('B', loc='left', fontsize=SIZE, weight='bold')
plt.xlabel(r'$\tau$ (Time)', fontsize=FontX)






ax1 = plt.subplot(N, M, 2)
initial = [2.9480480973709886, 10.6571221283170536, 1.2362714071535064]
data = series(1.4, 40.0, initial)
plt.plot(Time, data[:,0], '-r')
plt.plot(Time, data[:,1], '-b')

#plt.xlabel(r'Time', fontsize=FontX)
plt.legend([r'$\bar{P}_{T_1}$', r'$\bar{P}_{T_2}$'], ncol = 2, loc = 2, bbox_to_anchor = (0,1.07), frameon=False)
#plt.ylabel(r'Population', fontsize=FontY-7)
plt.yticks(fontsize=ticks_size)
plt.yticks([0.0, 5.0, 10.0, 15.0], fontsize=ticks_size);  plt.ylim([0.0,18.5])
plt.xticks([0, 25, 50, 75, 100], fontsize=ticks_size)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
#plt.title('D', loc='left', fontsize=SIZE, weight='bold')
plt.ylabel(r'Population', fontsize=FontY-4)




'''
ax1 = plt.subplot(N, M, 5)
for trial in range(2):
    if trial == 0: 
        initial = 0.379143418991746, 3.1784670911921524, 1.0760168782303208 #[ra.uniform(0,0.5), ra.uniform(3,3.5), ra.uniform(1.0,1.3)]; print (initial)
        data = series(2.15, 17.0, initial)
        plt.plot(Time, data[:,0], '-r')
        plt.plot(Time, data[:,1], '-b')
    else: 
        initial = [2.88914326776586, 0.6756181219296378, 1.0221693022270937]
        data = series(2.15, 17.0, initial)
        plt.plot(Time, data[:,0], '--r')
        plt.plot(Time, data[:,1], '--b')

#plt.legend([r'$\bar{P}_{T_1}$', r'$\bar{P}_{T_2}$'], ncol = 2, loc = 2, bbox_to_anchor = (0,1.07), frameon=False)
plt.xlabel(r'Time', fontsize=FontX)
plt.ylabel(r'Population', fontsize=FontY-4)
plt.yticks([0.0,1.5,3.0,4.5], fontsize=ticks_size);  plt.ylim([0.0,5.25])
plt.xticks([0, 25, 50, 75, 100], fontsize=ticks_size)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
plt.title('E', loc='left', fontsize=SIZE, weight='bold')
'''




ax1 = plt.subplot(N, M, 4)
for trial in range(2):
    if trial == 0: 
        initial = [0.5773933706066545, 3.806309042363984, 1.0698218641391566]
        data = series(2.15, 30.0, initial)
        plt.plot(Time, data[:,0], '-r')
        plt.plot(Time, data[:,1], '-b')
    else: 
        initial = [ra.uniform(0, 0.05), ra.uniform(4.4,4.5), ra.uniform(1.0,1.3)]; print (initial)
        data = series(2.15, 30.0, initial)
        plt.plot(Time, data[:,0], '--r')
        plt.plot(Time, data[:,1], '--b')

plt.legend([r'$\bar{P}_{T_1}$', r'$\bar{P}_{T_2}$'], ncol = 2, loc = 2, bbox_to_anchor = (0,1.07), frameon=False, fontsize=12)
plt.xlabel(r'$\tau$ (Time)', fontsize=FontX)
#plt.ylabel(r'Population', fontsize=FontY-7)
plt.yticks([0.0,2.0,4.0,6.0,7.1], fontsize=ticks_size); plt.ylim([-0.1,6.7])
plt.xticks([0, 25, 50, 75, 100], fontsize=ticks_size)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
#plt.title('E', loc='left', fontsize=SIZE, weight='bold')
plt.ylabel(r'Population', fontsize=FontY-4)





#plt.tight_layout()
plt.savefig('Figure8.pdf')
#plt.show()

####################################################################################################
