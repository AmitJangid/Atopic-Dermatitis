# Shi-Morio_bifurcation
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp, os, sys
import random as ra, time
from multiprocessing import Pool
from datetime import datetime
####################################################################################################
start_time = datetime.now()
print (start_time, 'STARTING TIME OF THE PROGRAM')
print ('####')

# Parameters
#mux=0.3 ; muy=0.2 ; muz=.4; muk=0.2
#p_a=0.01; p_c=0.01; p_e=0.01; p_g=0.01
#p_d=1.0; p_f=1.0; p_h=1.0; n=1.2 

# Order Th1, Th2, D, K Cells
Kx=1.00; Ky=1.00; Kz=1.00; Kk=1.00
p_q=0.01; p_r=0.01; p_s=0.01; p_p=0.01
p_v=0.6 ; p_w=0.4; p_u=0.4;

n=1.2; #p_b=20.0 

Itteration=int(sys.argv[1])
Suffix = 'b' # Suffix to save the file
PATH='/home/jangid/Desktop/root'

def FUN(INPUT):

	p_b,Ind,Suf = INPUT

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

	
	Iter = 20
	with open(os.path.join(PATH+'/il4/rootil4','il4{}_{}.dat'.format(Suf,Ind)), 'w') as f_series:
		for jj in range(Iter):
			# Initial guess of the roots! Most important for root finding

			Ini = np.array([ra.uniform(0.05,4.0),ra.uniform(0.05,4.0),ra.uniform(0.05,4.0),ra.uniform(0.05,4.0)])

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
				if np.any(Ini<0) == True: break
				else: 
					if ii == Iter1-1:
						if np.all(np.around(fun(Ini[0],Ini[1],Ini[2],Ini[3]), decimals=4) == 0.0): 
							f_series.write('%.5f %.5f %.5f %.5f %.5f\n'% (p_b, Ini[0],Ini[1],Ini[2],Ini[3]))

			if jj == Iter-1: print (Ind)
	

if __name__ == '__main__':
    N_Proces = 4
    Initial_X = np.linspace(0.0,4.0,Itteration) # Initial Conditions
    Suf = [Suffix]*len(Initial_X) # Suffix to save the file, Need to change every time
    Pool(N_Proces).map(FUN, zip(Initial_X, range(len(Initial_X)), Suf))

#print FUN((1.5,1000,0))

time.sleep(1)
# Finding files
DATA = []
for Ind in range(50): 
    dataa = np.genfromtxt(PATH+'/il4/rootil4/il4{}_{}.dat'.format(Suffix,Ind))
    if len(dataa) > 0: DATA.append(dataa)

# Merging all files into one file
with open(os.path.join(PATH+'/il4/dataR/il4{}.dat'.format(Suffix)), 'w') as f_series:
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

end_time = datetime.now()
print ('MAIN PROGRAM IS COMPLETED||Duration||H:M:S||{}'.format(end_time - start_time), '\n')
####################################################################################################

