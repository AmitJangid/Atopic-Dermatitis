
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp, os
import random as ra, time
from multiprocessing import Pool
from datetime import datetime
import sys
####################################################################################################
start_time = datetime.now()
print (start_time, 'Starts')
print ('####')

# Order Th1, Th2, D, K Cells
#Kx=1.00; Ky=1.00; Kz=1.00; Kk=1.00
#p_q=0.01; p_r=0.01; p_s=0.01; p_p=0.01
#p_v=0.6 ; p_w=0.4; p_u=0.4; n=2.0;

# Order Th1, Th2, D, K Cells
Kx=1.00; Ky=0.225; Kz=1.00; Kk=1.00
p_q=0.01; p_r=0.01; p_s=0.01; p_p=0.01
p_v=0.6 ; p_w=0.4; p_u=0.4; n=2.0;

POINTS =500
PATH = '.'


#Control
# --------
SaveS = 'Ky'
SaveN = Ky

ini, end = 0.5, 24.0; bista_ini, bista_end = 2.0, 6.8
para_ini = np.linspace(ini,bista_ini,int(POINTS*0.15))
para_dense = np.linspace(bista_ini, bista_end,int(POINTS*0.70))
para_end = np.linspace(bista_end,end,int(POINTS*0.15))

Initial_X = np.append(para_ini,para_dense)
Initial_X = np.append(Initial_X,para_end)

# --------    
    


# -----------------------------------------------------------------------------------------------------------------
def FUN(INPUT):

	p_b, Ind = INPUT    
	save_p = p_b
    
	# For derivation
	x,y,z,k = sp.symbols('x,y,z,k')

	# System of equations for differentiations
	f1 = (p_q) + (p_b/Kx) * ((1/(1+(x**n))) * (1/(1+(y**n))))                   - p_v * x       # For Th1 cell
	f2 = (p_r) + (p_b/Ky) * (((x**n)/(1+(x**n))) * ((y**n)/(1+(y**n))))         - p_w * y       # For Th2 cell
	f3 = (p_s) + (p_b/Kz) * (((x**n)/(1+(x**n))) * ((y**n)/(1+(y**n))))         - z             # For D cell
	f4 = (p_p) + (p_b/Kk) * ((1/(1+(x**n))) * (1/(1+(y**n))) * (1/(1+(z**n))))  - p_u * k       # For K cell

	# For function value and also for root precission
	def fun(x, y, z, k):
		f1 = (p_q) + (p_b/Kx) * ((1/(1+(x**n))) * (1/(1+(y**n))))                   - p_v * x       # For Th1 cell
		f2 = (p_r) + (p_b/Ky) * (((x**n)/(1+(x**n))) * ((y**n)/(1+(y**n))))         - p_w * y       # For Th2 cell
		f3 = (p_s) + (p_b/Kz) * (((x**n)/(1+(x**n))) * ((y**n)/(1+(y**n))))         - z             # For D cell
		f4 = (p_p) + (p_b/Kk) * ((1/(1+(x**n))) * (1/(1+(y**n))) * (1/(1+(z**n))))  - p_u * k       # For K cell
		return np.array([f1,f2,f3,f4], dtype='float')
	
	
	Iter = 5
	with open(os.path.join(PATH+'/junk/il4_{}.dat'.format(Ind)), 'w') as f_series:
		for jj in range(Iter):

			# Initial guess of the roots! Most important for root finding
			Ini = np.array([ra.uniform(0.01,7.0),ra.uniform(0.01,7.0),ra.uniform(0.01,7.0),ra.uniform(0.01,7.0)])
			
			# These initial conditions are for parameter 1 for bistability
			#if p_b > 10 and p_b < 30:
			#    Ini = np.array([ra.uniform(0.05,4.5),ra.uniform(2.0,4.0),ra.uniform(1.0,10.0)])
			#else:
			#    Ini = np.array([ra.uniform(0.1,10.0),ra.uniform(1.5,4.1),ra.uniform(1.0,10.0)])

			Iter1 = 20
			for ii in range(Iter1):
				SUB = [(x,Ini[0]),(y,Ini[1]),(z,Ini[2]),(k,Ini[3])]
				# Jacobian of the system
				JAC = np.array([[sp.diff(f1,x).subs(SUB),sp.diff(f1,y).subs(SUB),sp.diff(f1,z).subs(SUB),sp.diff(f1,k).subs(SUB)],\
								[sp.diff(f2,x).subs(SUB),sp.diff(f2,y).subs(SUB),sp.diff(f2,z).subs(SUB),sp.diff(f2,k).subs(SUB)],\
								[sp.diff(f3,x).subs(SUB),sp.diff(f3,y).subs(SUB),sp.diff(f3,z).subs(SUB),sp.diff(f3,k).subs(SUB)],\
								[sp.diff(f4,x).subs(SUB),sp.diff(f4,y).subs(SUB),sp.diff(f4,z).subs(SUB),sp.diff(f4,k).subs(SUB)]], dtype='float')

				# Iterations in the root finding
				INVE = np.linalg.inv(np.array(np.array(JAC, dtype='float')))
				PROD = np.dot(INVE, fun(Ini[0], Ini[1], Ini[2], Ini[3]))
				Ini = Ini - PROD
				
				if np.any(Ini<0) == True: 
				    break
				else: 
					if ii == Iter1-1:
						if np.all(np.around(fun(Ini[0],Ini[1],Ini[2],Ini[3]), decimals=4) == 0.0): 
							f_series.write('%.5f %.5f %.5f %.5f %.5f\n'% (save_p, Ini[0],Ini[1],Ini[2],Ini[3]))
			if jj == Iter-1: print (Ind)
    
	

if __name__ == '__main__':
    N_Proces = 30
    #ini, end = 0.01, 10.0; bista_ini, bista_end = 2.5,22
    #para_ini = np.linspace(ini,bista_ini,int(POINTS*0.20))
    #para_dense = np.linspace(bista_ini, bista_end,int(POINTS*0.60))
    #para_end = np.linspace(bista_end,end,int(POINTS*0.20))
    
    #Initial_X = np.append(para_ini,para_dense)
    #Initial_X = np.append(Initial_X,para_end)
    
    Initial_X = np.linspace(ini,end,POINTS) # Initial Conditions general case
    Pool(N_Proces).map(FUN, zip(Initial_X, range(len(Initial_X))))


#print (FUN(1.5,1000,0))

# -----------------------------------------------------------------------------------------------------------------
#
#
#
#
#
#
#
#
# -----------------------------------------------------------------------------------------------------------------

time.sleep(2)
# Finding files
DATA = []
for Ind in range(POINTS): 
    dataa = np.genfromtxt(PATH+'/junk/il4_{}.dat'.format(Ind))
    if len(dataa) > 0: DATA.append(dataa)

# Merging all files into one file
with open(os.path.join(PATH+'/data_1p/il4_b_{}_{}.dat'.format(SaveS, SaveN)), 'w') as f_series:
    for data, LE in zip(DATA, range(len(DATA))):
        if len(data) > 0:
            Data = np.unique(data,axis=0)  # After removing all same roots/data
            if len(np.shape(data)) > 1:   # Will check either number of different roors are there or not
                for da in Data:
                    f_series.write('%.5f %.5f %.5f %.5f %.5f\n'% (da[0],da[1],da[2],da[3],da[4]))		
            if len(np.shape(data)) == 1: # Will check if there is only one root
                f_series.write('%.5f %.5f %.5f %.5f %.5f\n'% (data[0],data[1],data[2],data[3],data[4]))		
        else: # If no root found then need again run the program
                print (LE)	

# -----------------------------------------------------------------------------------------------------------------








end_time = datetime.now()
print ('Ends|{}'.format(end_time - start_time), '\n')
####################################################################################################

