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

data_b_v_0 = np.genfromtxt(PATH+'/ifn_b_v_0.1.dat')
data_b_v_1 = np.genfromtxt(PATH+'/ifn_b_v_0.2.dat')
data_b_v_2 = np.genfromtxt(PATH+'/ifn_b_v_0.4.dat')

data_b_Kx_0 = np.genfromtxt(PATH+'/ifn_b_Kx_0.5.dat')
data_b_Kx_1 = np.genfromtxt(PATH+'/ifn_b_Kx_1.0.dat')
data_b_Kx_2 = np.genfromtxt(PATH+'/ifn_b_Kx_2.0.dat')

data_b_Ky_0 = np.genfromtxt(PATH+'/ifn_b_Ky_0.025.dat')
data_b_Ky_1 = np.genfromtxt(PATH+'/ifn_b_Ky_0.5.dat')
data_b_Ky_2 = np.genfromtxt(PATH+'/ifn_b_Ky_10.0.dat')

data_b_n_2  = np.genfromtxt(PATH+'/ifn_b_n_1.8.dat')
data_b_n_3 = np.genfromtxt(PATH+'/ifn_b_n_2.0.dat')
data_b_n_4 = np.genfromtxt(PATH+'/ifn_b_n_2.2.dat')


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
SIZE = 14; FontX = 20; FontY = 20; LW = 1.5; MS = 1.5; ticks_size=14.0; Legfs = 10
N, M = 4, 3

Title = ['A', 'B', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']





plt.figure(figsize=(3.8667*3.0,3.15*4.0))
plt.subplots_adjust(left=0.060, right=0.993, bottom=0.055, top=0.980, hspace=0.387, wspace=0.223)


########

ax3 = plt.subplot(N,M,1)
USb1 = []; Sb1L = []; Sb1U = []
for data in data_b_v_0:
    if data[1] > 0.241026 and data[1] < 0.823268: USb1.append(data)
    elif data[1] <= 0.241026: Sb1L.append(data)
    else: Sb1U.append(data)

plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,1], '--',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,2], '--',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,1], '-',color=colors['red'],linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,2], '-',color=colors['darkblue'],linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,1], '-',color=colors['red'], label=labelling_03[4], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,2], '-',color=colors['darkblue'], label=labelling_03[5], linewidth = LW, markersize=MS)
#plt.text(55, 13, r'v=0.1', fontsize=8.0)

X  = max(np.array(Sb1L)[:,0])
index = list(np.array(Sb1U)[:,0]).index(X)
Y  = np.array(Sb1L)[:,1][-1]
dY = np.array(Sb1U)[:,1][index] - Y
print ('Figure {}|'.format(Title[2]), np.around(X, decimals=1))
plt.plot([X for i in range(10)], [np.linspace(0,dY+Y,10)[i] for i in range(10)], '--k')
plt.text(X*1.75, 2.5, r' $\bar{b}_{_{IFN_{\gamma}}}$=7.5', fontsize=Legfs)
plt.legend(loc=2, ncol=3, fancybox=True, fontsize=Legfs, frameon=False)#, bbox_to_anchor=(0.5, 1.05))
plt.title(r'{}'.format(Title[2]), loc='left', fontsize=SIZE, weight='bold')
plt.title(r'${}$'.format(Para[3]), loc='right', fontsize=SIZE-2)
plt.xticks(fontsize=ticks_size); plt.yticks(fontsize=ticks_size)
#plt.xlabel(r'b$_{_{ifn-\gamma}}$', fontsize=FontX)
plt.ylabel(r'Steady state', fontsize=FontY)
plt.ylim([-0.2,20.0])




ax3 = plt.subplot(N,M,2)
USb1 = []; Sb1L = []; Sb1U = []
for data in data_b_v_1:
    if data[1] > 0.110262 and data[1] < 0.970342: USb1.append(data)
    elif data[1] <= 0.110262: Sb1L.append(data)
    else: Sb1U.append(data)

plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,1], '--',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,2], '--',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,1], '-',color=colors['red'],linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,2], '-',color=colors['darkblue'],linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,1], '-',color=colors['red'], label=labelling_03[2], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,2], '-',color=colors['darkblue'], label=labelling_03[3], linewidth = LW, markersize=MS)

X  = max(np.array(Sb1L)[:,0])
index = list(np.array(Sb1U)[:,0]).index(X)
Y  = np.array(Sb1L)[:,1][-1]
dY = np.array(Sb1U)[:,1][index] - Y
print ('Figure {}|'.format(Title[3]), np.around(X, decimals=1))
plt.plot([X for i in range(10)], [np.linspace(0,dY+Y,10)[i] for i in range(10)], '--k')
plt.text(X*1.5, 3.0, r'$\bar{b}_{_{IFN_{\gamma}}}$=27.2', fontsize=Legfs)
plt.title(r'{}'.format(Title[3]), loc='left', fontsize=SIZE, weight='bold')
plt.title(r'${}$'.format(Para[4]), loc='right', fontsize=SIZE-2)
plt.legend(loc=2, ncol=3, fancybox=True, fontsize=Legfs, frameon=False)#, bbox_to_anchor=(0.5, 1.05))
plt.yticks([0.0,5.0,10.0,15.0], fontsize=ticks_size); plt.xticks(fontsize=ticks_size)





ax3 = plt.subplot(N,M,3)

USb1 = []; Sb1L = []; Sb1U = []
for data in data_b_v_2:
    if data[1] > 0.0530851 and data[1] < 0.971813: USb1.append(data)
    elif data[1] <= 0.10530851: Sb1L.append(data)
    else: Sb1U.append(data)

plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,1], '--',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,2], '--',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,1], '-',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,2], '-',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,1], '-',color=colors['red'], label=labelling_03[0], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,2], '-',color=colors['darkblue'], label=labelling_03[1], linewidth = LW, markersize=MS)

X  = max(np.array(Sb1L)[:,0])
index = list(np.array(Sb1U)[:,0]).index(X)
Y  = np.array(Sb1L)[:,1][-1]
dY = np.array(Sb1U)[:,1][index] - Y
print ('Figure {}|'.format(Title[4]), np.around(X, decimals=1))
plt.plot([106 for i in range(10)], [np.linspace(0,dY+Y,10)[i] for i in range(10)], '--k')
plt.text(X*0.65, 0.65, r'$\bar{b}_{_{IFN_{\gamma}}}$=106', fontsize=Legfs)
plt.title(r'{}'.format(Title[4]), loc='left', fontsize=SIZE, weight='bold')
plt.title(r'${}$'.format(Para[5]), loc='right', fontsize=SIZE-2)
plt.legend(loc=2, ncol=3, fancybox=True, fontsize=Legfs, frameon=False)#, bbox_to_anchor=(0.5, 1.05))
plt.xticks(fontsize=ticks_size); plt.yticks(fontsize=ticks_size)
#plt.ylabel(r'Steady state', fontsize=FontY)











########

plt.subplot(N,M,4)
USb1 = []; Sb1L = []; Sb1U = []
for data in data_b_Kx_0:
    if data[1] > 0.100912 and data[1] < 0.881977: USb1.append(data)
    elif data[1] <= 0.100912: Sb1L.append(data)
    else: Sb1U.append(data)

plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,1], '--',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,2], '--',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,1], '-',color=colors['red'],linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,2], '-',color=colors['darkblue'],linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,1], '-',color=colors['red'], label=labelling_04[4], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,2], '-',color=colors['darkblue'], label=labelling_04[5], linewidth = LW, markersize=MS)


X  = max(np.array(Sb1L)[:,0])
index = list(np.array(Sb1U)[:,0]).index(X)
Y  = np.array(Sb1L)[:,1][-1]
dY = np.array(Sb1U)[:,1][index] - Y
print ('Figure {}|'.format(Title[5]), np.around(X, decimals=1))
plt.plot([X for i in range(10)], [np.linspace(0,dY+Y,10)[i] for i in range(10)], '--k')
plt.text(X*1.15, 0.5, r' $\bar{b}_{_{IFN_{\gamma}}}$=13.9', fontsize=Legfs)
plt.legend(loc=2, ncol=3, fancybox=True, fontsize=Legfs, frameon=False)#, bbox_to_anchor=(0.5, 1.05))
plt.title(r'{}'.format(Title[5]), loc='left', fontsize=SIZE, weight='bold')
plt.title(r'${}$'.format(Para[6]), loc='right', fontsize=SIZE-2)
plt.xticks(fontsize=ticks_size); plt.yticks(fontsize=ticks_size)
#plt.xlabel(r'$b_{_{IFN_{\gamma}}}$', fontsize=FontX)
plt.ylabel(r'Steady state', fontsize=FontY)
plt.ylim([-0.2,11.0])




plt.subplot(N,M,5)
USb1 = []; Sb1L = []; Sb1U = []
for data in data_b_Kx_1:
    if data[1] > 0.112983 and data[1] < 0.913957: USb1.append(data)
    elif data[1] <= 0.112983: Sb1L.append(data)
    else: Sb1U.append(data)

plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,1], '--',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,2], '--',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,1], '-',color=colors['red'],linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,2], '-',color=colors['darkblue'],linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,1], '-',color=colors['red'], label=labelling_04[2], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,2], '-',color=colors['darkblue'], label=labelling_04[3], linewidth = LW, markersize=MS)


X  = max(np.array(Sb1L)[:,0])
index = list(np.array(Sb1U)[:,0]).index(X)
Y  = np.array(Sb1L)[:,1][-1]
dY = np.array(Sb1U)[:,1][index] - Y
print ('Figure {}|'.format(Title[6]), np.around(X, decimals=1))
plt.plot([X for i in range(10)], [np.linspace(0,dY+Y,10)[i] for i in range(10)], '--k')
plt.text(X*1.05, 0.5, r'$\bar{b}_{_{IFN_{\gamma}}}$=27.4', fontsize=Legfs)
plt.legend(loc=2, ncol=3, fancybox=True, fontsize=Legfs, frameon=False)#, bbox_to_anchor=(0.5, 1.05))
plt.title(r'{}'.format(Title[6]), loc='left', fontsize=SIZE, weight='bold')
plt.title(r'${}$'.format(Para[7]), loc='right', fontsize=SIZE-2)
plt.xticks(fontsize=ticks_size); plt.yticks(fontsize=ticks_size)
#plt.xlabel(r'$b_{_{IFN_{\gamma}}}$', fontsize=FontX)
#plt.ylabel(r'Steady state', fontsize=FontY)
plt.ylim([-0.2,11.0])



plt.subplot(N,M,6)

USb1 = []; Sb1L = []; Sb1U = []
for data in data_b_Kx_2:
    if data[1] > 0.100038 and data[1] < 0.934875: USb1.append(data)
    elif data[1] <= 0.100038: Sb1L.append(data)
    else: Sb1U.append(data)

plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,1], '--',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,2], '--',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,1], '-',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,2], '-',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,1], '-',color=colors['red'], label=labelling_04[0], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,2], '-',color=colors['darkblue'], label=labelling_04[1], linewidth = LW, markersize=MS)

X  = max(np.array(Sb1L)[:,0])
index = list(np.array(Sb1U)[:,0]).index(X)
Y  = np.array(Sb1L)[:,1][-1]
dY = np.array(Sb1U)[:,1][index] - Y
print ('Figure {}|'.format(Title[7]), np.around(X, decimals=1))
plt.plot([X for i in range(10)], [np.linspace(0,dY+Y,10)[i] for i in range(10)], '--k')
plt.text(X*1.05, 0.5, r'$\bar{b}_{_{IFN_{\gamma}}}$=53.5', fontsize=Legfs)
plt.legend(loc=2, ncol=3, fancybox=True, fontsize=Legfs, frameon=False)#, bbox_to_anchor=(0.5, 1.05))
plt.title(r'{}'.format(Title[7]), loc='left', fontsize=SIZE, weight='bold')
plt.title(r'${}$'.format(Para[8]), loc='right', fontsize=SIZE-2)
plt.xticks(fontsize=ticks_size); plt.yticks(fontsize=ticks_size)
#plt.xlabel(r'$b_{_{IFN_{\gamma}}}$', fontsize=FontX)
#plt.ylabel(r'Steady state', fontsize=FontY)
plt.ylim([-0.2,11.0])












########

plt.subplot(N,M,7)
USb1 = []; Sb1L = []; Sb1U = []
for data in data_b_Ky_0:
    if data[1] > 0.108339 and data[1] < 0.949966: USb1.append(data)
    elif data[1] <= 0.108339: Sb1L.append(data)
    else: Sb1U.append(data)

plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,1], '--',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,2], '--',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,1], '-',color=colors['red'],linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,2], '-',color=colors['darkblue'],linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,1], '-',color=colors['red'], label=labelling_05[4], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,2], '-',color=colors['darkblue'], label=labelling_05[5], linewidth = LW, markersize=MS)

X  = max(np.array(Sb1L)[:,0])
index = list(np.array(Sb1U)[:,0]).index(X)
Y  = np.array(Sb1L)[:,1][-1]
dY = np.array(Sb1U)[:,1][index] - Y;
print ('Figure {}|'.format(Title[8]), np.around(X, decimals=1))
plt.plot([X for i in range(10)], [np.linspace(0,dY+Y,10)[i] for i in range(10)], '--k')
plt.text(X*1.05, 0.75, r'$\bar{b}_{_{IFN_{\gamma}}}$=26.4', fontsize=Legfs)
plt.legend(loc=2, ncol=3, fancybox=True, fontsize=Legfs-0.5, frameon=False)#, bbox_to_anchor=(0.5, 1.05))
plt.title(r'{}'.format(Title[8]), loc='left', fontsize=SIZE, weight='bold')
plt.title(r'${}$'.format(Para[9]), loc='right', fontsize=SIZE-2)
plt.xticks(fontsize=ticks_size); plt.yticks(fontsize=ticks_size)
#plt.xlabel(r'$b_{_{IFN_{\gamma}}}$', fontsize=FontX)
plt.ylabel(r'Steady state', fontsize=FontY)
plt.ylim([-0.1,22.0])



plt.subplot(N,M,8)
USb1 = []; Sb1L = []; Sb1U = []
for data in data_b_Ky_1:
    if data[1] > 0.120016 and data[1] < 0.895958: USb1.append(data)
    elif data[1] <= 0.120016: Sb1L.append(data)
    else: Sb1U.append(data)

plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,1], '--',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,2], '--',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,1], '-',color=colors['red'],linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,2], '-',color=colors['darkblue'],linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,1], '-',color=colors['red'], label=labelling_05[2], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,2], '-',color=colors['darkblue'], label=labelling_05[3], linewidth = LW, markersize=MS)

X  = max(np.array(Sb1L)[:,0])
index = list(np.array(Sb1U)[:,0]).index(X)
Y  = np.array(Sb1L)[:,1][-1]
dY = np.array(Sb1U)[:,1][index] - Y;
print ('Figure {}|'.format(Title[9]), np.around(X, decimals=1))
plt.plot([X for i in range(10)], [np.linspace(0,dY+Y,10)[i] for i in range(10)], '--k')
plt.text(X*1.05, 0.5, r'$\bar{b}_{_{IFN_{\gamma}}}$=27.5', fontsize=Legfs)
plt.legend(loc=2, ncol=3, fancybox=True, fontsize=Legfs, frameon=False)#, bbox_to_anchor=(0.5, 1.05))
plt.title(r'{}'.format(Title[9]), loc='left', fontsize=SIZE, weight='bold')
plt.title(r'${}$'.format(Para[10]), loc='right', fontsize=SIZE-2)
plt.xticks(fontsize=ticks_size); plt.yticks(fontsize=ticks_size)
#plt.xlabel(r'$b_{_{IFN_{\gamma}}}$', fontsize=FontX)
plt.ylim([-0.1,8.0])




plt.subplot(N,M,9)

USb1 = []; Sb1L = []; Sb1U = []
for data in data_b_Ky_2:
    if data[1] > 0.109377 and data[1] < 0.872513: USb1.append(data)
    elif data[1] <= 0.109377: Sb1L.append(data)
    else: Sb1U.append(data)

plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,1], '--',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,2], '--',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,1], '-',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,2], '-',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,1], '-',color=colors['red'], label=labelling_05[0], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,2], '-',color=colors['darkblue'], label=labelling_05[1], linewidth = LW, markersize=MS)

X  = max(np.array(Sb1L)[:,0])
index = list(np.array(Sb1U)[:,0]).index(X)
Y  = np.array(Sb1L)[:,1][-1]
dY = np.array(Sb1U)[:,1][index] - Y;
print ('Figure {}|'.format(Title[10]), np.around(X, decimals=1))
plt.plot([X for i in range(10)], [np.linspace(0,dY+Y,10)[i] for i in range(10)], '--k')
plt.text(X*0.90, 5.5, r'$\bar{b}_{_{IFN_{\gamma}}}$=32.8', fontsize=Legfs)
plt.legend(loc=2, ncol=3, fancybox=True, fontsize=Legfs, frameon=False)#, bbox_to_anchor=(0.5, 1.05))
plt.title(r'{}'.format(Title[10]), loc='left', fontsize=SIZE, weight='bold')
plt.title(r'${}$'.format(Para[11]), loc='right', fontsize=SIZE-2)
plt.xticks(fontsize=ticks_size); plt.yticks(fontsize=ticks_size)
#plt.xlabel(r'$b_{_{IFN_{\gamma}}}$', fontsize=FontX)
plt.ylim([-0.1,8.0])














########

ax6 = plt.subplot(N,M,10)
USb1 = []; Sb1L = []; Sb1U = []
for data in data_b_n_2:
    if data[1] > 0.113466 and data[1] < 0.749685: USb1.append(data)
    elif data[1] <= 0.113466: Sb1L.append(data)
    else: Sb1U.append(data)

plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,1], '--',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,2], '--',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,1], '-',color=colors['red'],linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,2], '-',color=colors['darkblue'],linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,1], '-',color=colors['red'], label=labelling_02[4], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,2], '-',color=colors['darkblue'], label=labelling_02[5], linewidth = LW, markersize=MS)

X  = max(np.array(USb1)[:,0])
print ('Figure {}|'.format(Title[11]), np.around(X, decimals=1))
plt.plot([X for i in range(10)], [np.linspace(0,2.5,10)[i] for i in range(10)], '--k')
plt.text(X*1.15, 0.5, r'$\bar{b}_{_{IFN_{\gamma}}}$=13.0', fontsize=Legfs)
plt.legend(loc=2, ncol=3, fancybox=True, fontsize=Legfs, frameon=False)#, bbox_to_anchor=(0.5, 1.05))
plt.title(r'{}'.format(Title[11]), loc='left', fontsize=SIZE, weight='bold')
plt.title(r'${}$'.format(Para[12]), loc='right', fontsize=SIZE-2)
plt.yticks([0.0,5.0,10.0,15.0], fontsize=ticks_size); plt.xticks(fontsize=ticks_size)
plt.xlabel(r'$b_{_{IFN_{\gamma}}}$', fontsize=FontX)
plt.ylabel(r'Steady state', fontsize=FontY)



ax6 = plt.subplot(N,M,11)
USb1 = []; Sb1L = []; Sb1U = []
for data in data_b_n_3:
    if data[1] > 0.103691 and data[1] < 0.890171: USb1.append(data)
    elif data[1] <= 0.103691: Sb1L.append(data)
    else: Sb1U.append(data)

plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,1], '--',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,2], '-',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,1], '-',color=colors['red'],linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,2], '-',color=colors['darkblue'],linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,1], '-',color=colors['red'], label=labelling_02[2], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,2], '-',color=colors['darkblue'], label=labelling_02[3], linewidth = LW, markersize=MS)

X  = max(np.array(Sb1L)[:,0])
index = list(np.array(Sb1U)[:,0]).index(X)
Y  = np.array(Sb1L)[:,1][-1]
dY = np.array(Sb1U)[:,1][index] - Y
print ('Figure {}|'.format(Title[12]), np.around(X, decimals=1))
plt.plot([X for i in range(10)], [np.linspace(0,dY+Y,10)[i] for i in range(10)], '--k')
plt.text(X*1.15, 0.5, r'$\bar{b}_{_{IFN_{\gamma}}}$=27.5', fontsize=Legfs)
plt.legend(loc=2, ncol=3, fancybox=True, fontsize=Legfs, frameon=False)#, bbox_to_anchor=(0.5, 1.05))
plt.title(r'{}'.format(Title[12]), loc='left', fontsize=SIZE, weight='bold')
plt.title(r'${}$'.format(Para[13]), loc='right', fontsize=SIZE-2)
plt.xticks(fontsize=ticks_size); plt.yticks(fontsize=ticks_size)
plt.xlabel(r'$b_{_{IFN_{\gamma}}}$', fontsize=FontX)



ax6 = plt.subplot(N,M,12)

USb1 = []; Sb1L = []; Sb1U = []
for data in data_b_n_4: 
	if data[1] > 0.0926073 and data[1] < 1.00659: USb1.append(data)
	elif data[1] <= 0.0926073: Sb1L.append(data)
	else: Sb1U.append(data)

plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,1], '--',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,2], '--',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,1], '-',color=colors['red'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,2], '-',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,1], '-',color=colors['red'], label=labelling_02[0], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,2], '-',color=colors['darkblue'], label=labelling_02[1], linewidth = LW, markersize=MS)

X  = max(np.array(Sb1L)[:,0])
index = list(np.array(Sb1U)[:,0]).index(X)
Y  = np.array(Sb1L)[:,1][-1]
dY = np.array(Sb1U)[:,1][index] - Y
print ('Figure {}|'.format(Title[13]), np.around(X, decimals=1))
plt.plot([X for i in range(10)], [np.linspace(0,dY+Y,10)[i] for i in range(10)], '--k')
plt.text(X*1.05, 0.5, r'$\bar{b}_{_{IFN_{\gamma}}}$=57.7', fontsize=Legfs)
plt.legend(loc=2, ncol=3, fancybox=True, fontsize=Legfs, frameon=False)#, bbox_to_anchor=(0.5, 1.05))
plt.title(r'{}'.format(Title[13]), loc='left', fontsize=SIZE, weight='bold')
plt.title(r'${}$'.format(Para[14]), loc='right', fontsize=SIZE-2)
plt.xticks(fontsize=ticks_size); plt.yticks(fontsize=ticks_size)
plt.xlabel(r'$b_{_{IFN_{\gamma}}}$', fontsize=FontX)
#plt.ylabel(r'Steady state', fontsize=FontY)










########
#plt.tight_layout()
#plt.savefig('ifn_b_bet_paper.pdf'); #plt.savefig('ifn_b_bet_paper.png')
plt.savefig('Figure3.pdf'); #plt.savefig('ifn_b_bet_paper.png')
#plt.show()

################################################################################################################################
