#
#
##########################################
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import random, os, time
from datetime import datetime
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from matplotlib import colors as mcolors

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
##############################################################################################################################
start_time = datetime.now()

PATH = './script/data_1p'

data_p = np.genfromtxt(PATH+'/ifn_p_s1.dat')
data_u = np.genfromtxt(PATH+'/ifn_u_s1.dat')
data_Kz = np.genfromtxt(PATH+'/ifn_Kz_s1.dat')

data_q = np.genfromtxt(PATH+'/ifn_q_s1.dat')
data_v = np.genfromtxt(PATH+'/ifn_v_s1.dat')
data_Kx = np.genfromtxt(PATH+'/ifn_Kx_s1.dat')

data_r = np.genfromtxt(PATH+'/ifn_r_s1.dat')
data_w = np.genfromtxt(PATH+'/ifn_w_s1.dat')
data_Ky = np.genfromtxt(PATH+'/ifn_Ky_s1.dat')


#colors = plt.cm.jet(np.linspace(0, 1, 4)); 
labelling_01 = [r'$\bar{P}_{_{T_1}}$', r'$\bar{P}_{_{T_2}}$']

# Tick Label size and Axis label size
SIZE = 17; FontX = 20; FontY = 20; LW = 1.5; MS = 1.5; ticks_size=14.0
N, M = 3, 3







plt.figure(figsize=(3.8667*3.0,3.15*3.0))



########
ax1 = plt.subplot(N,M,1)

plt.plot(data_p[:,0],data_p[:,1], '-',color=colors['darkred'], linewidth = LW, markersize=MS)
plt.plot(data_p[:,0],data_p[:,2], '-',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.title('A', loc='left', fontsize=SIZE, weight='bold')
plt.legend([r'$\bar{P}_{T_1}$', r'$\bar{P}_{T_2}$'], loc=2, ncol=2, fancybox=True, fontsize=8, bbox_to_anchor=(0.55, 1.18))
plt.xticks(fontsize=ticks_size); plt.yticks(fontsize=ticks_size)
plt.ylabel(r'Steady state', fontsize=FontY)
plt.xlabel(r'$p$', fontsize=FontX)

########
ax2 = plt.subplot(N,M,2)

plt.plot(data_u[:,0],data_u[:,1], '-',color=colors['darkred'], linewidth = LW, markersize=MS)
plt.plot(data_u[:,0],data_u[:,2], '-',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.title('B', loc='left', fontsize=SIZE, weight='bold')
plt.legend([r'$\bar{P}_{T_1}$', r'$\bar{P}_{T_2}$'], loc=2, ncol=2, fancybox=True, fontsize=8, bbox_to_anchor=(0.55, 1.18))
plt.xticks(fontsize=ticks_size); plt.yticks(fontsize=ticks_size)
plt.xlabel(r'$u$', fontsize=FontX)

########
ax3 = plt.subplot(N,M,3)

plt.plot(data_Kz[:,0],data_Kz[:,1], '-',color=colors['darkred'], linewidth = LW, markersize=MS)
plt.plot(data_Kz[:,0],data_Kz[:,2], '-',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.title('C', loc='left', fontsize=SIZE, weight='bold')
plt.legend([r'$\bar{P}_{T_1}$', r'$\bar{P}_{T_2}$'], loc=2, ncol=2, fancybox=True, fontsize=8, bbox_to_anchor=(0.55, 1.18))
plt.xticks(fontsize=ticks_size); plt.yticks(fontsize=ticks_size)
plt.xlabel(r'$k_{K}$', fontsize=FontX)





########
ax4 = plt.subplot(N,M,4)

plt.plot(data_q[:,0],data_q[:,1], '-',color=colors['darkred'], linewidth = LW, markersize=MS)
plt.plot(data_q[:,0],data_q[:,2], '-',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.title('D', loc='left', fontsize=SIZE, weight='bold')
plt.legend([r'$\bar{P}_{T_1}$', r'$\bar{P}_{T_2}$'], loc=2, ncol=2, fancybox=True, fontsize=8, bbox_to_anchor=(0.55, 1.18))
plt.xticks(fontsize=ticks_size); plt.yticks(fontsize=ticks_size)
plt.xlabel(r'$q$', fontsize=FontX)
plt.ylabel(r'Steady state', fontsize=FontY)

########
ax5 = plt.subplot(N,M,5)

USb1 = []; Sb1L = []; Sb1U = []
for data in data_v:
    if data[1] > 0.277258 and data[1] < 0.806079: USb1.append(data)
    elif data[1] <= 0.277258: Sb1L.append(data)
    else: Sb1U.append(data)
print (len(USb1))

plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,1], '-',color=colors['darkred'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,2], '-',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,1], '--',color=colors['darkred'], linewidth = LW, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,2], '--',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,1], '-',color=colors['darkred'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,2], '-',color=colors['darkblue'], linewidth = LW, markersize=MS)

#plt.plot(data_v[:,0],data_v[:,1], '.',color=colors['darkred'], linewidth = LW, markersize=MS)
#plt.plot(data_v[:,0],data_v[:,2], '.',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.title('E', loc='left', fontsize=SIZE, weight='bold')
plt.legend([r'$\bar{P}_{T_1}$', r'$\bar{P}_{T_2}$'], loc=2, ncol=2, fancybox=True, fontsize=8, bbox_to_anchor=(0.55, 1.18))
plt.xticks(fontsize=ticks_size); plt.yticks(fontsize=ticks_size)
plt.xlabel(r'$v$', fontsize=FontX)


########
ax6 = plt.subplot(N,M,6)

USb1 = []; Sb1L = []; Sb1U = []
for data in data_Kx:
    if data[1] > 0.108996 and data[1] < 0.881122: USb1.append(data)
    elif data[1] <= 0.108996: Sb1L.append(data)
    else: Sb1U.append(data)
print (len(USb1))

plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,1], '-',color=colors['darkred'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1U)[:,0],np.array(Sb1U)[:,2], '-',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,1], '--',color=colors['darkred'], linewidth = LW, markersize=MS)
plt.plot(np.array(USb1)[:,0],np.array(USb1)[:,2], '--',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,1], '-',color=colors['darkred'], linewidth = LW, markersize=MS)
plt.plot(np.array(Sb1L)[:,0],np.array(Sb1L)[:,2], '-',color=colors['darkblue'], linewidth = LW, markersize=MS)


plt.title('F', loc='left', fontsize=SIZE, weight='bold')
plt.legend([r'$\bar{P}_{T_1}$', r'$\bar{P}_{T_2}$'], loc=2, ncol=2, fancybox=True, fontsize=8, bbox_to_anchor=(0.55, 1.18))
plt.xticks(fontsize=ticks_size); plt.yticks(fontsize=ticks_size)
plt.xlabel(r'$k_{T_1}$', fontsize=FontX)





########
ax7 = plt.subplot(N,M,7)

plt.plot(data_r[:,0],data_r[:,1], '-',color=colors['darkred'], linewidth = LW, markersize=MS)
plt.plot(data_r[:,0],data_r[:,2], '-',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.title('G', loc='left', fontsize=SIZE, weight='bold')
plt.legend([r'$\bar{P}_{T_1}$', r'$\bar{P}_{T_2}$'], loc=2, ncol=2, fancybox=True, fontsize=8, bbox_to_anchor=(0.55, 1.18))
plt.xticks(fontsize=ticks_size); plt.yticks(fontsize=ticks_size)
plt.xlabel(r'$r$', fontsize=FontX)
plt.ylabel(r'Steady state', fontsize=FontY)


########
ax8 = plt.subplot(N,M,8)

plt.plot(data_w[:,0],data_w[:,1], '-',color=colors['darkred'], linewidth = LW, markersize=MS)
plt.plot(data_w[:,0],data_w[:,2], '-',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.title('H', loc='left', fontsize=SIZE, weight='bold')
plt.legend([r'$\bar{P}_{T_1}$', r'$\bar{P}_{T_2}$'], loc=2, ncol=2, fancybox=True, fontsize=8, bbox_to_anchor=(0.55, 1.18))
plt.xticks(fontsize=ticks_size); plt.yticks(fontsize=ticks_size)
plt.xlabel(r'$w$', fontsize=FontX)


########
ax9 = plt.subplot(N,M,9)

plt.plot(data_Ky[:,0],data_Ky[:,1], '-',color=colors['darkred'], linewidth = LW, markersize=MS)
plt.plot(data_Ky[:,0],data_Ky[:,2], '-',color=colors['darkblue'], linewidth = LW, markersize=MS)
plt.title('I', loc='left', fontsize=SIZE, weight='bold')
plt.legend([r'$\bar{P}_{T_1}$', r'$\bar{P}_{T_2}$'], loc=2, ncol=2, fancybox=True, fontsize=8, bbox_to_anchor=(0.55, 1.18))
plt.xticks(fontsize=ticks_size); plt.yticks(fontsize=ticks_size)
plt.xlabel(r'$k_{T_2}$', fontsize=FontX)





########
plt.tight_layout()
plt.savefig('fig_s1.pdf')#; plt.savefig('ifn_all.png')
plt.show()

end_time = datetime.now()
print ('PROGRAM IS COMPLETED||TOTAL TIME TAKEN||Duration||H:M:S||{}'.format(end_time - start_time), '\n')
################################################################################################################################
