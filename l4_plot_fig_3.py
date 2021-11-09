import numpy as np, sys
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
Kx=1.0; Ky=1.0; Kz=1.00; Kk=1.00
p_q=0.01; p_r=0.01; p_s=0.01; p_p=0.01
p_v=0.6 ; p_w=0.4; p_u=0.4; 
n=2.0

Time = np.linspace(0, 120, 15000)
# TNF-alpha network 
def series(trials):
    for k in range(trials):
        def ato(Var, t):
            x, y, z, k = Var
            f1 = (p_q) + (p_b/Kx) * ((1/(1+(x**n))) * (1/(1+(y**n))) * (1/(1+(k**n))))                  - p_v * x               # For Th1 Cell
            f2 = (p_r) + (p_b/Ky) * (((x**n)/(1+(x**n))) * ((y**n)/(1+(y**n))) * ((k**n)/(1+(k**n))))   - p_w * y               # For Th2 Cell
            f3 = (p_s) + (p_b/Kz) * (((x**n)/(1+(x**n))) * ((y**n)/(1+(y**n))) * ((k**n)/(1+(k**n))))   - z                     # For D Cell
            f4 = (p_p) + (p_b/Kk) * ((1/(1+(x**n))) * (1/(1+(y**n))) * (1/(1+(k**n))))                  - p_u * k               # For K Cell
            return [f1, f2, f3, f4]

        Var0 = [ra.uniform(0.01,2), ra.uniform(2,4.0), ra.uniform(1.0,1.3), ra.uniform(1.0,1.3)]

        sol = odeint(ato, Var0, Time)

    return sol

#
#
#
#
#
#







PATH = './scripts/'

# Tick Label size and Axis label size
SIZE = 17; FontX = 20; FontY = 20; 
LWseries = 0.75; LWbifurcation = 1.5; MS = 1.5; ticks_size=14.0
N, M = 4, 2

plt.figure(figsize=(3.8667*3.0,3.15*1.3))
plt.subplots_adjust(left=0.058, right=0.993, bottom=0.16, top=0.925, hspace=0.387, wspace=0.223)


##########
plt.subplot2grid((N, M), (0,0), colspan=1, rowspan=4)

data_01 = np.genfromtxt(PATH+'data_2p/il4_n_b_bistable.dat'); 
data_02 = np.genfromtxt(PATH+'data_2p/il4_n_b_transition.dat'); 
data_02_dict = dict(data_02); data_02_compare = []
for key in data_01[:,0]: data_02_compare.append(data_02_dict[key])

#plt.plot(data_02[35:,0], data_02[35:,1], '--k', markersize=5.0);

plt.fill_between(data_01[:,0],data_01[:,1], data_01[:,2], color=colors['darkgrey'])
plt.fill_between(data_01[:,0],data_01[:,1], color=colors['red'])
plt.fill_between(data_02[:29,0],data_02[:29,1], color=colors['red'])
plt.fill_between(data_02[:30,0],data_02[:30,1], 12.0, color=colors['mediumblue'])

plt.fill_between(data_01[:7,0],data_01[:7,2], 12.0, color=colors['mediumblue'])
plt.fill_between(data_01[:,0],data_01[:,2], data_02_compare, where=data_01[:,2]<data_02_compare, color=colors['red'])
plt.fill_between(data_01[6:,0],data_01[6:,2], 12.0, color=colors['mediumblue'])

plt.title('A', loc='left', fontsize=SIZE, weight='bold')
plt.xticks([1.0, 1.4, 1.8, 2.2], fontsize=ticks_size); plt.yticks([0,3.0,6.0,9.0], fontsize=ticks_size)
plt.xlabel(r'$n$', fontsize=FontX)
plt.ylabel(r'$\bar{b}_{_{IL4}}$', fontsize=FontY)
plt.axis([1.0,2.2,0,9.0])


plt.text(1.2, 7.0, 'Acute', fontsize=15, color='w')
plt.text(1.4, 0.55, 'Chronic', fontsize=15, color='w')
#plt.text(1.74, 8.5, 'Bistable (acute)', fontsize=8, color='k', rotation=15)
plt.text(1.9, 4.0, 'Bistable', fontsize=15, color='k', rotation=45)




##########
plt.subplot2grid((N, M), (0,1), colspan=1, rowspan=4)

data_11 = np.genfromtxt(PATH+'/data_2p/il4_kT1_b_bistable.dat'); 
data_12 = np.genfromtxt(PATH+'/data_2p/il4_kT1_b_transition.dat'); 

#plt.plot(data_12[:,0], data_12[:,1], '--k', markersize=5.0);
plt.fill_between(data_11[:,0],data_11[:,1], color=colors['red'])
plt.fill_between(data_11[:,0],data_11[:,2], 9.0, color=colors['mediumblue'])
plt.fill_between(data_11[:,0],data_11[:,1], data_11[:,2], color=colors['darkgrey'])

plt.title('B', loc='left', fontsize=SIZE, weight='bold')
plt.xticks([0, 1.0, 2.0, 3.0, 4.0], fontsize=ticks_size); plt.yticks([0.0,3.0, 6.0, 9.0], fontsize=ticks_size)
plt.xlabel(r'$k_{T_1}$', fontsize=FontX)
plt.ylabel(r'$\bar{b}_{_{IL4}}$', fontsize=FontY)
plt.axis([0.05,4.0,0,9])


plt.text(1.0, 0.5, 'Chronic', fontsize=15, color='w')
plt.text(1.0, 7.0, 'Acute', fontsize=15, color='w')
#plt.text(1.05, 14.0, 'Bistable (acute)', fontsize=10, color='k', rotation=17)
plt.text(1.0, 4.5, 'Bistable', fontsize=10, color='k', rotation=15)


#plt.tight_layout()
#plt.savefig('{}.pdf'.format(sys.argv[0][5:-3])); #plt.savefig('ifn_b_bet_paper.png')
plt.savefig('Figure7.pdf'); #plt.savefig('ifn_b_bet_paper.png')
#plt.show()

####################################################################################################

