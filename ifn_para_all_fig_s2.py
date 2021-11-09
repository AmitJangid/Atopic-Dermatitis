# Shi-Morio_bifurcation
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp, os
import random as ra, time
from multiprocessing import Pool
from datetime import datetime
####################################################################################################
start_time = datetime.now()
print (start_time, 'STARTING TIME OF THE PROGRAM')
print ('####')

# Parameters
#mux=0.3 ; muy=0.2 ; muz=0.2; d=0.2
#p_a=0.01; p_c=0.01; p_d=1.0
#p_e=0.01; p_f=1.0;  n=2.25

# Standard
#Kx=1.1;    Ky=0.5;   Kz=2.00
#p_q=0.01;  p_r=1.0;  p_p=0.3
#p_v=0.3 ;  p_w=0.8;  p_u = 0.3
#d=0.2;     n=1.3;    p_b = 5.0

# Standard New
#Kx=1.0;    Ky=0.5;   Kz=2.00
#p_q=0.01;  p_r=1.0;  p_p=1.0
#p_v=0.2 ;  p_w=0.8;  p_u = 0.3
#d=0.2;     n=2.0

# Order Th1, Th2, D, K Cells
Kx=1.0;    Ky=0.5;   Kz=2.0
p_q=0.01;  p_r=1.0;  p_p=1.0
p_v=0.2 ;  p_w=0.8;  p_u = 0.3
d=0.2;     n=2.0;    p_b = 20.0


POINTS = 500
PATH = '.'

Parameter_Save = 'w'
def FUN(INPUT):

	p_w, Ind =  INPUT
	
	# This is the parameter which is varying
	Par_V = p_w 

	# For derivation
	x,y,z = sp.symbols('x,y,z');


    # f1 corresponds to Th1, f2 corresponds to Th2, and f3 corresponds to K cell equations
	# System of equations for differentiations
	f1 = p_q + (p_b/Kx) * ((x**n)/(1+(x**n))) * ((y**n)/(1+(y**n))) * ((d**n)/(1+(d**n))) - p_v * x
	f2 = p_r + (p_b/Ky) * (1/(1+(x**n))) * (1/(1+(y**n))) * (1/(1+(d**n))) - p_w * y
	f3 = p_p + (p_b/Kz) * (1/(1+(x**n))) * (1/(1+(y**n))) * (1/(1+(d**n))) - p_u * z

	# For function value and also for root precission
	def fun(x, y, z):
		f1 = p_q + (p_b/Kx) * ((x**n)/(1+(x**n))) * ((y**n)/(1+(y**n))) * ((d**n)/(1+(d**n))) - p_v * x
		f2 = p_r + (p_b/Ky) * (1/(1+(x**n))) * (1/(1+(y**n))) * (1/(1+(d**n))) - p_w * y
		f3 = p_p + (p_b/Kz) * (1/(1+(x**n))) * (1/(1+(y**n))) * (1/(1+(d**n))) - p_u * z
		return np.array([f1,f2,f3], dtype='float')

	
	Iter = 20
	with open(os.path.join(PATH+'/junk/ifn_{}.dat'.format(Ind)), 'w') as f_series:
		for jj in range(Iter):
			# Initial guess of the roots! Most important for root finding

			# These initial conditions are for parameter 1 for bistability
			if p_b > 5.0 and p_b < 11.0:
			    Ini = np.array([ra.uniform(0.0,2.0),ra.uniform(1.5,3.0),ra.uniform(1.0,3.0)])
			else:
			    Ini = np.array([ra.uniform(0.1,5.0),ra.uniform(1.0,4.1),ra.uniform(1.0,3.0)])

			Iter1 = 30
			for ii in range(Iter1):
				SUB = [(x,Ini[0]),(y,Ini[1]),(z,Ini[2])]
				# Jacobian of the system
				JAC = np.array([[sp.diff(f1,x).subs(SUB),sp.diff(f1,y).subs(SUB),sp.diff(f1,z).subs(SUB)],\
								[sp.diff(f2,x).subs(SUB),sp.diff(f2,y).subs(SUB),sp.diff(f2,z).subs(SUB)],\
								[sp.diff(f3,x).subs(SUB),sp.diff(f3,y).subs(SUB),sp.diff(f3,z).subs(SUB)]], dtype='float')

				INVE = np.linalg.inv(np.array(np.array(JAC, dtype='float')))
				PROD = np.dot(INVE, fun(Ini[0], Ini[1], Ini[2]))
				Ini = Ini - PROD
				
				if np.any(Ini<0) == True: 
					break
				else: 
					if np.all(np.around(fun(Ini[0],Ini[1],Ini[2]), decimals=4) == 0.0): 
						f_series.write('%.5f %.5f %.5f %.5f\n'% (Par_V,Ini[0],Ini[1],Ini[2]))
						break

			if jj == Iter-1: print (Ind)
	

if __name__ == '__main__':
    N_Proces = 12
    ini, end = 0.2, 10.0; bista_ini, bista_end = 2.5, 7.5
    para_ini = np.linspace(ini,bista_ini,int(POINTS*0.15))
    para_dense = np.linspace(bista_ini, bista_end,int(POINTS*0.70))
    para_end = np.linspace(bista_end,end,int(POINTS*0.15))
    
    Initial_X = np.append(para_ini,para_dense)
    Initial_X = np.append(Initial_X,para_end)
    
    Initial_X = np.linspace(ini, end,POINTS) # Initial Conditions general case
    Pool(N_Proces).map(FUN, zip(Initial_X, range(len(Initial_X))))

#print FUN((1.5,1000,0))



time.sleep(2)
# Finding files
DATA = [];
for Ind in range(POINTS):
    dataa = np.genfromtxt(PATH+'/junk/ifn_{}.dat'.format(Ind))
    if len(dataa) > 0: DATA.append(dataa)



# Merging all files into one file
print ('Merging all files into one output file')

with open(os.path.join(PATH+'/data_1p/ifn_{}_s2.dat'.format(Parameter_Save)), 'w') as f_series:
    for data, LE in zip(DATA, range(len(DATA))):
        if len(data) > 0:
            Data = np.unique(data,axis=0)  # After removing all same roots/data
            if len(np.shape(data)) > 1:   # Will check either number of different roors are there or not
                for da in Data:
                    f_series.write('%.5f %.5f %.5f %.5f\n'% (da[0],da[1],da[2],da[3]))		
            if len(np.shape(data)) == 1: # Will check if there is only one root
                f_series.write('%.5f %.5f %.5f %.5f\n'% (data[0],data[1],data[2],data[3]))		
        else: # If no root found then again run the program
                print (LE)	

end_time = datetime.now()
print ('MAIN PROGRAM IS COMPLETED||Duration||H:M:S||{}'.format(end_time - start_time), '\n')
####################################################################################################

