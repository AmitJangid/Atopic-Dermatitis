#
#
##########################################
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
##############################################################################################################################


PATH = './script/data_1p'


data_b_n_0 = np.genfromtxt(PATH+'/ifn_b_n_1.4.dat')
data_b_n_1 = np.genfromtxt(PATH+'/ifn_b_nn_2.0.dat')


labelling_02 = [r'$\bar{P}_{T_1}$', r'$\bar{P}_{T_2}$', \
                r'$\bar{P}_{T_1}$', r'$\bar{P}_{T_2}$', \
                r'$\bar{P}_{T_1}$', r'$\bar{P}_{T_2}$']
                
labelling_03 = [r'$\bar{P}_{T_1}$', r'$\bar{P}_{T_2}$', \
                r'$\bar{P}_{T_1}$', r'$\bar{P}_{T_2}$', \
                r'$\bar{P}_{T_1}$', r'$\bar{P}_{T_2}$']
                
labelling_04 = [r'$\bar{P}_{T_1}$', r'$\bar{P}_{T_2}$', \
                r'$\bar{P}_{T_1}$', r'$\bar{P}_{T_2}$', \
                r'$\bar{P}_{T_1}$', r'$\bar{P}_{T_2}$']
                
labelling_05 = [r'$\bar{P}_{T_1}$', r'$\bar{P}_{T_2}$', \
                r'$\bar{P}_{T_1}$', r'$\bar{P}_{T_2}$', \
                r'$\bar{P}_{T_1}$', r'$\bar{P}_{T_2}$']
               
        
Para = [r'n=1.4', r'n=2.0', '', \
        'v=0.1', 'v=0.2', 'v=0.4', \
        'k_{T_1}=0.5', 'k_{T_1}=1.0', 'k_{T_1}=2.0', \
        'k_{T_2}=0.025', 'k_{T_2}=0.5', 'k_{T_2}=10.0', \
        'n=1.8', 'n=2.0', 'n=2.2']


# Tick Label size and Axis label size
SIZE = 14; FontX = 20; FontY = 20; LW = 1.5; MS = 1.5; ticks_size=14.0; Legfs = 9
N, M = 1, 2

Title = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']





plt.figure(figsize=(3.8667*3.0,3.15*1.2))
plt.subplots_adjust(left=0.050, right=0.993, bottom=0.205, top=0.925, hspace=0.387, wspace=0.223)

########
ax1 = plt.subplot(N,M,1)

plt.plot(data_b_n_0[:,0],data_b_n_0[:,1], '-',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(data_b_n_0[:,0],data_b_n_0[:,2], '-',color=colors['darkblue'], linewidth = LW, markersize=MS)

for x, y1, y2 in zip(data_b_n_0[:,0], data_b_n_0[:,1], data_b_n_0[:,2]): 
    if y1 > y2: y_plot = x; print ('Figure {}|'.format(Title[0]), np.around(x, decimals=1)); break
plt.plot([y_plot for i in range(10)], [np.linspace(0,3,10)[i] for i in range(10)], '--k')
plt.text(y_plot*1.05, 0.5, r'$\bar{b}_{_{IFN_{\gamma}}}$=8.1', fontsize=Legfs+3)
plt.title(r'{}'.format(Title[0]), loc='left', fontsize=SIZE, weight='bold')
#plt.title(r'${}$'.format(Para[0]), loc='right', fontsize=SIZE-2)
plt.legend([r'$\bar{P}_{T_1}$', r'$\bar{P}_{T_2}$'], loc=2, ncol=2, fancybox=True, fontsize=Legfs+2, frameon=False)
plt.xticks([0.0, 5.0, 10.0, 15.0], fontsize=ticks_size); plt.yticks([0, 1, 2, 3, 4, 5], fontsize=ticks_size)
plt.ylabel(r'Steady state', fontsize=FontY)
plt.xlabel(r'$\bar{b}_{_{IFN_{\gamma}}}$', fontsize=FontX)


########
ax2 = plt.subplot(N,M,2)

USb1 = []; Sb1L = []; Sb1U = []
for data in data_b_n_1:
    if data[1] > 0.103691 and data[1] < 0.890171: USb1.append(data)
    elif data[1] <= 0.103691: Sb1L.append(data)
    else: Sb1U.append(data)

plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,1], '--',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,2], '--',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,1], '-',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,2], '-',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,1], '-',color=colors['red'], label=labelling_02[2], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,2], '-',color=colors['darkblue'], label=labelling_02[3], linewidth = LW, markersize=MS)

X  = max(np.array(Sb1L)[:,0])
index = list(np.array(Sb1U)[:,0]).index(X)
Y  = np.array(Sb1L)[:,1][-1]
dY = np.array(Sb1U)[:,1][index] - Y
print ('Figure {}|'.format(Title[1]), np.around(X, decimals=1))
plt.plot([X for i in range(10)], [np.linspace(0,2.5,10)[i] for i in range(10)], '--k')
plt.text(X*1.05, 2.5, r'$\bar{b}_{_{IFN_{\gamma}}}$=27.6', fontsize=Legfs+3)
plt.legend(loc=2, fancybox=True, ncol=2, fontsize=Legfs+2, frameon=False)#, bbox_to_anchor=(0.5, 1.05))
plt.title(r'{}'.format(Title[1]), loc='left', fontsize=SIZE, weight='bold')
#plt.title(r'${}$'.format(Para[1]), loc='right', fontsize=SIZE-2)
plt.xticks([0, 10, 20, 30, 40], fontsize=ticks_size); plt.yticks([0, 2, 4, 6], fontsize=ticks_size)
plt.xlabel(r'$\bar{b}_{_{IFN_{\gamma}}}$', fontsize=FontX)
plt.text(8.4, 0.8, r'SN$_2$')
plt.text(28.5, 0.0, r'SN$_1$')


########
#plt.tight_layout()
#plt.savefig('ifn_b_bet_paper.pdf'); #plt.savefig('ifn_b_bet_paper.png')
plt.savefig('Figure2.pdf'); #plt.savefig('ifn_b_bet_paper.png')
plt.show()

################################################################################################################################
