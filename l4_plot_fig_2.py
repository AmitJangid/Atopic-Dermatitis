#
#
##########################################
import numpy as np, sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
##############################################################################################################################


PATH = './scripts/data_1p'

data_b_n_0 = np.genfromtxt(PATH+'/il4_b_n_1.3.dat')
data_b_n_1 = np.genfromtxt(PATH+'/il4_b_n_1.9.dat')


data_b_v_0 = np.genfromtxt(PATH+'/il4_b_v_0.4.dat')
data_b_v_1 = np.genfromtxt(PATH+'/il4_b_v_1.2.dat')
data_b_v_2 = np.genfromtxt(PATH+'/il4_b_v_3.6.dat')

data_b_Kx_0 = np.genfromtxt(PATH+'/il4_b_Kx_0.25.dat')
data_b_Kx_1 = np.genfromtxt(PATH+'/il4_b_Kx_1.0.dat')
data_b_Kx_2 = np.genfromtxt(PATH+'/il4_b_Kx_4.0.dat')

data_b_Ky_0 = np.genfromtxt(PATH+'/il4_b_Ky_0.225.dat')
data_b_Ky_1 = np.genfromtxt(PATH+'/il4_b_Ky_0.9.dat')
data_b_Ky_2 = np.genfromtxt(PATH+'/il4_b_Ky_3.6.dat')

data_b_n_2 = np.genfromtxt(PATH+'/il4_b_n_1.6.dat')
data_b_n_3 = np.genfromtxt(PATH+'/il4_b_n_2.0.dat')
data_b_n_4 = np.genfromtxt(PATH+'/il4_b_n_2.4.dat')



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
               
        
Para = [r'n=1.3', r'n=1.9', '', \
        'v=0.4', 'v=1.2', 'v=3.6', \
        'k_{T_1}=0.25', 'k_{T_1}=1.0', 'k_{T_1}=4.0', \
        'k_{T_2}=0.225', 'k_{T_2}=0.9', 'k_{T_2}=3.6', \
        'n=1.6', 'n=2.0', 'n=2.4']


# Tick Label size and Axis label size
SIZE = 14; FontX = 20; FontY = 20; LW = 1.5; MS = 1.5; ticks_size=14.0; Legfs = 10
N, M = 4, 3

Title = ['', '', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']





plt.figure(figsize=(3.8667*3.0,3.15*4.0))
plt.subplots_adjust(left=0.050, right=0.993, bottom=0.06, top=0.980, hspace=0.387, wspace=0.223)


########

ax3 = plt.subplot(N,M,1)
USb1 = []; Sb1L = []; Sb1U = []
for data in data_b_v_0:
    if data[2] > 0.052 and data[2] < 0.70: USb1.append(data)
    elif data[2] <= 0.052: Sb1L.append(data)
    else: Sb1U.append(data)

plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,1], '--',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,2], '--',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,1], '-',color=colors['red'],linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,2], '-',color=colors['darkblue'],linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,1], '-',color=colors['red'], label=labelling_03[0], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,2], '-',color=colors['darkblue'], label=labelling_03[1], linewidth = LW, markersize=MS)

X  = max(np.array(Sb1L)[:,0])
print ('Figure {}|'.format(Title[2]), np.around(X, decimals=1))
plt.plot([X for i in range(10)], [np.linspace(0,1.75,10)[i] for i in range(10)], '--k')
plt.text(X*1.15, 2.5, r'$\bar{b}_{_{IL4}}=4.8$', fontsize=Legfs)
plt.legend(loc=2, ncol=3, fancybox=True, fontsize=Legfs, frameon=False)#, bbox_to_anchor=(0.5, 1.05))
plt.title(r'{}'.format(Title[2]), loc='left', fontsize=SIZE, weight='bold')
plt.title(r'${}$'.format(Para[3]), loc='right', fontsize=SIZE-2)
plt.xticks([0.0,2.5,5.0,7.5,10.0], fontsize=ticks_size); plt.yticks([0.0,2.0,4.0,6.0,8.0], fontsize=ticks_size)
#plt.xlabel(r'b$_{_{ifn-\gamma}}$', fontsize=FontX)
plt.ylabel(r'Steady state', fontsize=FontY)
plt.xlim([0.0,10.2])



ax3 = plt.subplot(N,M,2)
USb1 = []; Sb1L = []; Sb1U = []
for data in data_b_v_1:
    if data[2] > 0.055 and data[2] < 0.614: USb1.append(data)
    elif data[2] <= 0.055: Sb1L.append(data)
    else: Sb1U.append(data)

plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,1], '--',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,2], '--',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,1], '-',color=colors['red'],linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,2], '-',color=colors['darkblue'],linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,1], '-',color=colors['red'], label=labelling_03[2], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,2], '-',color=colors['darkblue'], label=labelling_03[3], linewidth = LW, markersize=MS)


X  = max(np.array(Sb1L)[:,0])
print ('Figure {}|'.format(Title[3]), np.around(X, decimals=1))
plt.plot([X for i in range(10)], [np.linspace(0,1.8,10)[i] for i in range(10)], '--k')
plt.text(X*1.25, 1.5, r'$\bar{b}_{_{IL4}}=5.8$', fontsize=Legfs)
plt.text(115, 15, r'v=0.2', fontsize=8.0)
plt.title(r'{}'.format(Title[3]), loc='left', fontsize=SIZE, weight='bold')
plt.title(r'${}$'.format(Para[4]), loc='right', fontsize=SIZE-2)
plt.legend(loc=2, ncol=3, fancybox=True, fontsize=Legfs, frameon=False)#, bbox_to_anchor=(0.5, 1.05))
plt.yticks([0.0,2.0,4.0], fontsize=ticks_size); plt.xticks([0.0,2.5,5.0,7.5,10.0], fontsize=ticks_size)




ax3 = plt.subplot(N,M,3)

USb1 = []; Sb1L = []; Sb1U = []
for data in data_b_v_2:
    if data[2] > 0.050 and data[2] < 0.6096: USb1.append(data)
    elif data[2] <= 0.050: Sb1L.append(data)
    else: Sb1U.append(data)

plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,1], '--',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,2], '--',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,1], '-',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,2], '-',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,1], '-',color=colors['red'], label=labelling_03[4], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,2], '-',color=colors['darkblue'], label=labelling_03[5], linewidth = LW, markersize=MS)


X  = max(np.array(Sb1L)[:,0])
print ('Figure {}|'.format(Title[4]), np.around(X, decimals=1))
plt.plot([X for i in range(10)], [np.linspace(0,1,10)[i] for i in range(10)], '--k')
plt.text(8.0, 1.0, r'$\bar{b}_{_{IL4}}=7.7$', fontsize=Legfs)
plt.title(r'{}'.format(Title[4]), loc='left', fontsize=SIZE, weight='bold')
plt.title(r'${}$'.format(Para[5]), loc='right', fontsize=SIZE-2)
plt.legend(loc=2, ncol=3, fancybox=True, fontsize=Legfs, frameon=False)#, bbox_to_anchor=(0.5, 1.05))
plt.xticks([0.0,2.5,5.0,7.5,10.0], fontsize=ticks_size); plt.yticks([0.0,1.0,2.0,3.0], fontsize=ticks_size)
#plt.ylabel(r'Steady state', fontsize=FontY)











########

plt.subplot(N,M,4)
USb1 = []; Sb1L = []; Sb1U = []
for data in data_b_Kx_0:
    if data[2] > 0.054 and data[2] < 0.7: USb1.append(data)
    elif data[2] <= 0.054: Sb1L.append(data)
    else: Sb1U.append(data)

plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,1], '--',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,2], '--',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,1], '-',color=colors['red'],linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,2], '-',color=colors['darkblue'],linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,1], '-',color=colors['red'], label=labelling_04[4], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,2], '-',color=colors['darkblue'], label=labelling_04[5], linewidth = LW, markersize=MS)

X  = max(np.array(Sb1L)[:,0])
print ('Figure {}|'.format(Title[5]), np.around(X, decimals=1))
plt.plot([X for i in range(10)], [np.linspace(0,2.0,10)[i] for i in range(10)], '--k')
plt.text(X*1.15, 1.5, r'$\bar{b}_{_{IL4}}=4.1$', fontsize=Legfs)
plt.legend(loc=2, ncol=3, fancybox=True, fontsize=Legfs, frameon=False)#, bbox_to_anchor=(0.5, 1.05))
plt.title(r'{}'.format(Title[5]), loc='left', fontsize=SIZE, weight='bold')
plt.title(r'${}$'.format(Para[6]), loc='right', fontsize=SIZE-2)
plt.xticks([0.0,2.5,5.0,7.5,10.0], fontsize=ticks_size); plt.yticks([0, 2.0, 4.0, 6.0, 8.0], fontsize=ticks_size)
#plt.xlabel(r'$b_{_{IFN_{\gamma}}}$', fontsize=FontX)
plt.ylabel(r'Steady state', fontsize=FontY)




plt.subplot(N,M,5)
USb1 = []; Sb1L = []; Sb1U = []
for data in data_b_Kx_1:
    if data[2] > 0.06 and data[2] < 0.7: USb1.append(data)
    elif data[2] <= 0.06: Sb1L.append(data)
    else: Sb1U.append(data)

plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,1], '--',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,2], '--',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,1], '-',color=colors['red'],linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,2], '-',color=colors['darkblue'],linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,1], '-',color=colors['red'], label=labelling_04[2], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,2], '-',color=colors['darkblue'], label=labelling_04[3], linewidth = LW, markersize=MS)

X  = max(np.array(Sb1L)[:,0])
print ('Figure {}|'.format(Title[6]), np.around(X, decimals=1))
plt.plot([X for i in range(10)], [np.linspace(0,1.7,10)[i] for i in range(10)], '--k')
plt.text(X*1.05, 1.5, r'$\bar{b}_{_{IL4}}=5.1$', fontsize=Legfs)
plt.legend(loc=2, ncol=3, fancybox=True, fontsize=Legfs, frameon=False)#, bbox_to_anchor=(0.5, 1.05))
plt.title(r'{}'.format(Title[6]), loc='left', fontsize=SIZE, weight='bold')
plt.title(r'${}$'.format(Para[7]), loc='right', fontsize=SIZE-2)
plt.xticks([0.0,2.5,5.0,7.5,10.0], fontsize=ticks_size); plt.yticks([0, 2.0, 4.0, 6.0], fontsize=ticks_size)
#plt.xlabel(r'$b_{_{IFN_{\gamma}}}$', fontsize=FontX)
#plt.ylabel(r'Steady state', fontsize=FontY)



plt.subplot(N,M,6)

USb1 = []; Sb1L = []; Sb1U = []
for data in data_b_Kx_2:
    if data[2] > 0.051 and data[2] < 0.55: USb1.append(data)
    elif data[2] <= 0.051: Sb1L.append(data)
    else: Sb1U.append(data)

plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,1], '--',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,2], '--',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,1], '-',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,2], '-',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,1], '-',color=colors['red'], label=labelling_04[0], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,2], '-',color=colors['darkblue'], label=labelling_04[1], linewidth = LW, markersize=MS)

X  = max(np.array(Sb1L)[:,0])
print ('Figure {}|'.format(Title[7]), np.around(X, decimals=1))
plt.plot([X for i in range(10)], [np.linspace(0,1,10)[i] for i in range(10)], '--k')
plt.text(X*1.05, 1.0, r'$\bar{b}_{_{IL4}}=6.8$', fontsize=Legfs)
plt.legend(loc=2, ncol=3, fancybox=True, fontsize=Legfs, frameon=False)#, bbox_to_anchor=(0.5, 1.05))
plt.title(r'{}'.format(Title[7]), loc='left', fontsize=SIZE, weight='bold')
plt.title(r'${}$'.format(Para[8]), loc='right', fontsize=SIZE-2)
plt.xticks([0.0,2.5,5.0,7.5,10.0], fontsize=ticks_size); plt.yticks([0, 1.0, 2.0, 3.0], fontsize=ticks_size)
#plt.xlabel(r'$b_{_{IFN_{\gamma}}}$', fontsize=FontX)
#plt.ylabel(r'Steady state', fontsize=FontY)












########

plt.subplot(N,M,7)
USb1 = []; Sb1L = []; Sb1U = []
for data in data_b_Ky_0:
    if data[2] > 0.06 and data[2] < 0.66: USb1.append(data)
    elif data[2] <= 0.06: Sb1L.append(data)
    else: Sb1U.append(data)

plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,1], '--',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,2], '--',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,1], '-',color=colors['red'],linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,2], '-',color=colors['darkblue'],linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,1], '-',color=colors['red'], label=labelling_05[4], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,2], '-',color=colors['darkblue'], label=labelling_05[5], linewidth = LW, markersize=MS)


X  = max(np.array(Sb1L)[:,0])
print ('Figure {}|'.format(Title[8]), np.around(X, decimals=1))
plt.plot([X for i in range(10)], [np.linspace(0,1.7,10)[i] for i in range(10)], '--k')
plt.text(X*1.5, 1.0, r'$\bar{b}_{_{IL4}}=2.1$', fontsize=Legfs)
plt.legend(loc=2, ncol=3, fancybox=True, fontsize=Legfs-0.5, frameon=False)#, bbox_to_anchor=(0.5, 1.05))
plt.title(r'{}'.format(Title[8]), loc='left', fontsize=SIZE, weight='bold')
plt.title(r'${}$'.format(Para[9]), loc='right', fontsize=SIZE-2)
plt.xticks([0, 2.0, 4.0, 6.0, 8.0], fontsize=ticks_size); plt.yticks(fontsize=ticks_size)
#plt.xlabel(r'$b_{_{IFN_{\gamma}}}$', fontsize=FontX)
plt.ylabel(r'Steady state', fontsize=FontY)
plt.axis([-0.1,8.0,-0.1,6.0])



plt.subplot(N,M,8)
USb1 = []; Sb1L = []; Sb1U = []
for data in data_b_Ky_1:
    if data[2] > 0.05 and data[2] < 0.65: USb1.append(data)
    elif data[2] <= 0.05: Sb1L.append(data)
    else: Sb1U.append(data)

plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,1], '--',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,2], '--',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,1], '-',color=colors['red'],linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,2], '-',color=colors['darkblue'],linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,1], '-',color=colors['red'], label=labelling_05[2], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,2], '-',color=colors['darkblue'], label=labelling_05[3], linewidth = LW, markersize=MS)


X  = max(np.array(Sb1L)[:,0])
print ('Figure {}|'.format(Title[9]), np.around(X, decimals=1))
plt.plot([X for i in range(10)], [np.linspace(0,1.8,10)[i] for i in range(10)], '--k')
plt.text(X*1.25, 1.5, r'$\bar{b}_{_{IL4}}=5.1$', fontsize=Legfs)
plt.legend(loc=2, ncol=3, fancybox=True, fontsize=Legfs, frameon=False)#, bbox_to_anchor=(0.5, 1.05))
plt.title(r'{}'.format(Title[9]), loc='left', fontsize=SIZE, weight='bold')
plt.title(r'${}$'.format(Para[10]), loc='right', fontsize=SIZE-2)
plt.xticks([0, 6.0, 12.0, 18.0, 24.0], fontsize=ticks_size); plt.yticks([0.0, 3.0, 6.0, 9.0], fontsize=ticks_size)
#plt.xlabel(r'$b_{_{IFN_{\gamma}}}$', fontsize=FontX)




plt.subplot(N,M,9)

USb1 = []; Sb1L = []; Sb1U = []
for data in data_b_Ky_2:
    if data[2] > 0.05 and data[2] < 0.70: USb1.append(data)
    elif data[2] <= 0.05: Sb1L.append(data)
    else: Sb1U.append(data)

plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,1], '--',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,2], '--',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,1], '-',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,2], '-',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,1], '-',color=colors['red'], label=labelling_05[0], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,2], '-',color=colors['darkblue'], label=labelling_05[1], linewidth = LW, markersize=MS)


X  = max(np.array(Sb1L)[:,0])
print ('Figure {}|'.format(Title[10]), np.around(X, decimals=1))
plt.plot([X for i in range(10)], [np.linspace(0,2,10)[i] for i in range(10)], '--k')
plt.text(18, 1.75, r'$\bar{b}_{_{IL4}}=15.9$', fontsize=Legfs)
plt.legend(loc=2, ncol=3, fancybox=True, fontsize=Legfs, frameon=False)#, bbox_to_anchor=(0.5, 1.05))
plt.title(r'{}'.format(Title[10]), loc='left', fontsize=SIZE, weight='bold')
plt.title(r'${}$'.format(Para[11]), loc='right', fontsize=SIZE-2)
plt.xticks([0, 6.0, 12.0, 18.0, 24.0], fontsize=ticks_size); plt.yticks(fontsize=ticks_size)
#plt.xlabel(r'$b_{_{IFN_{\gamma}}}$', fontsize=FontX)














########

ax6 = plt.subplot(N,M,10)
USb1 = []; Sb1L = []; Sb1U = []
for data in data_b_n_2[5:]:
    if data[2] > 0.0525 and data[2] < 0.608: USb1.append(data)
    elif data[2] <= 0.0525: Sb1L.append(data)
    else: Sb1U.append(data)

plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,1], '--',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,2], '--',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,1], '-',color=colors['red'],linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,2], '-',color=colors['darkblue'],linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,1], '-',color=colors['red'], label=labelling_02[0], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,2], '-',color=colors['darkblue'], label=labelling_02[1], linewidth = LW, markersize=MS)

X  = max(np.array(USb1)[:,0])
print ('Figure {}|'.format(Title[11]), np.around(X, decimals=1))
plt.plot([X for i in range(10)], [np.linspace(0,1,10)[i] for i in range(10)], '--k')
plt.text(4.0, 2.0, r'$\bar{b}_{_{IL4}}=2.1$', fontsize=Legfs)
plt.legend(loc=2, ncol=3, fancybox=True, fontsize=Legfs, frameon=False)#, bbox_to_anchor=(0.5, 1.05))
plt.title(r'{}'.format(Title[11]), loc='left', fontsize=SIZE, weight='bold')
plt.title(r'${}$'.format(Para[12]), loc='right', fontsize=SIZE-2)
plt.yticks([0.0,2.0, 4.0, 6.0, 8.0], fontsize=ticks_size); plt.xticks([0.0, 4.0, 8.0, 12.0], fontsize=ticks_size)
plt.xlabel(r'$\bar{b}_{_{IL4}}$', fontsize=FontX)
plt.ylabel(r'Steady state', fontsize=FontY)
plt.xlim([-0.1,12.0])



ax6 = plt.subplot(N,M,11)
USb1 = []; Sb1L = []; Sb1U = []
for data in data_b_n_3[5:]:
    if data[2] > 0.0525 and data[2] < 0.608: USb1.append(data)
    elif data[2] <= 0.0525: Sb1L.append(data)
    else: Sb1U.append(data)

plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,1], '--',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,2], '--',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,1], '-',color=colors['red'],linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,2], '-',color=colors['darkblue'],linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,1], '-',color=colors['red'], label=labelling_02[2], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,2], '-',color=colors['darkblue'], label=labelling_02[3], linewidth = LW, markersize=MS)


X  = max(np.array(Sb1L)[:,0])
print ('Figure {}|'.format(Title[12]), np.around(X, decimals=1))
plt.plot([X for i in range(10)], [np.linspace(0,1.5,10)[i] for i in range(10)], '--k')
plt.text(X*1.15, 1.5, r'$\bar{b}_{_{IL4}}=5.1$', fontsize=Legfs)
plt.legend(loc=2, ncol=3, fancybox=True, fontsize=Legfs, frameon=False)#, bbox_to_anchor=(0.5, 1.05))
plt.title(r'{}'.format(Title[12]), loc='left', fontsize=SIZE, weight='bold')
plt.title(r'${}$'.format(Para[13]), loc='right', fontsize=SIZE-2)
plt.xticks([0.0, 6.0, 12.0, 18.0], fontsize=ticks_size); plt.yticks(fontsize=ticks_size)
plt.xlabel(r'$\bar{b}_{_{IL4}}$', fontsize=FontX)



ax6 = plt.subplot(N,M,12)

USb1 = []; Sb1L = []; Sb1U = []
for data in data_b_n_4[5:]: 
    if data[2] > 0.0525 and data[2] < 0.608: USb1.append(data)
    elif data[2] <= 0.0525: Sb1L.append(data)
    else: Sb1U.append(data)

plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,1], '--',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,2], '--',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,1], '-',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,2], '-',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,1], '-',color=colors['red'], label=labelling_02[4], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,2], '-',color=colors['darkblue'], label=labelling_02[5], linewidth = LW, markersize=MS)


X  = max(np.array(Sb1L)[:,0])
print ('Figure {}|'.format(Title[13]), np.around(X, decimals=1))
plt.plot([X for i in range(10)], [np.linspace(0,1.5,10)[i] for i in range(10)], '--k')
plt.text(X*.85, 3.0, r'$\bar{b}_{_{IL4}}=15.2$', fontsize=Legfs)
plt.legend(loc=2, ncol=3, fancybox=True, fontsize=Legfs, frameon=False)#, bbox_to_anchor=(0.5, 1.05))
plt.title(r'{}'.format(Title[13]), loc='left', fontsize=SIZE, weight='bold')
plt.title(r'${}$'.format(Para[14]), loc='right', fontsize=SIZE-2)
plt.xticks([0.0, 6.0, 12.0, 18.0], fontsize=ticks_size); plt.yticks([0.0, 2.0, 4.0, 6.0], fontsize=ticks_size)
plt.xlabel(r'$\bar{b}_{_{IL4}}$', fontsize=FontX)
#plt.ylabel(r'Steady state', fontsize=FontY)








########
#plt.tight_layout()
#plt.savefig('ifn_b_bet_paper.pdf'); #plt.savefig('ifn_b_bet_paper.png')
#plt.savefig('{}.pdf'.format(sys.argv[0][5:-3])); #plt.savefig('ifn_b_bet_paper.png')
plt.savefig('Figure5.pdf'); #plt.savefig('ifn_b_bet_paper.png')
#plt.show()

################################################################################################################################
