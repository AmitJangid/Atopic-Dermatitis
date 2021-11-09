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


# Order Th1, Th2, D, K Cells
#Kx=1.00; Ky=1.00; Kz=1.00; Kk=1.00
#p_q=0.01; p_r=0.01; p_s=0.01; p_p=0.01
#p_v=0.6 ; p_w=0.4; p_u=0.4; n=2.0;

# Order Th1, Th2, D, K Cells
Kx=1.00; Ky=1.0; Kz=1.00; Kk=1.00
p_q=0.01; p_r=0.01; p_s=0.01; p_p=0.01
p_v=0.6 ; p_w=0.4; p_u=0.4; n=2.0;


POINTS = 100; times_n = 4
PATH = '.'

def FUN(INPUT):

	n, Ind_1 =  INPUT

	for p_b, Ind_2 in zip(np.linspace(0,20.0,int(POINTS*times_n)), range(int(POINTS*times_n))):    
    
	    # For derivation
	    x,y,z,k = sp.symbols('x,y,z,k');


        # f1, f2, f3, f4 = Th1, Th2, D, K
	    # System of equations for differentiations
	    f1 = (p_q) + (p_b/Kx) * ((1/(1+(x**n))) * (1/(1+(y**n))))                   - p_v * x       # For Th1 cell
	    f2 = (p_r) + (p_b/Ky) * (((x**n)/(1+(x**n))) * ((y**n)/(1+(y**n))))         - p_w * y       # For Th2 cell
	    f3 = (p_s) + (p_b/Kz) * (((x**n)/(1+(x**n))) * ((y**n)/(1+(y**n))))         - z             # For D cell
	    f4 = (p_p) + (p_b/Kk) * ((1/(1+(x**n))) * (1/(1+(y**n))) * (1/(1+(z**n))))  - p_u * k       # For K cell
	
	    # For function value and also for root precission
	    def fun(x, y, z, k):
	        # System of equations for differentiations
	        f1 = (p_q) + (p_b/Kx) * ((1/(1+(x**n))) * (1/(1+(y**n))))                   - p_v * x       # For Th1 cell
	        f2 = (p_r) + (p_b/Ky) * (((x**n)/(1+(x**n))) * ((y**n)/(1+(y**n))))         - p_w * y       # For Th2 cell
	        f3 = (p_s) + (p_b/Kz) * (((x**n)/(1+(x**n))) * ((y**n)/(1+(y**n))))         - z             # For D cell
	        f4 = (p_p) + (p_b/Kk) * ((1/(1+(x**n))) * (1/(1+(y**n))) * (1/(1+(z**n))))  - p_u * k       # For K cell
	        return np.array([f1,f2,f3,f4], dtype='float')

	    
	    Iter = 20
	    with open(os.path.join(PATH+'/junk/il4_{}_{}.dat'.format(Ind_1, Ind_2)), 'w') as f_series:
		    for jj in range(Iter):
			    # Initial guess of the roots! 

			    #if p_b > 10 and p_b < 30:
			    #    Ini = np.array([ra.uniform(0.01,3.0),ra.uniform(0.01,3.0),ra.uniform(0.01,3.0),ra.uniform(0.01,3.0)])
			    #else:
			    #    Ini = np.array([ra.uniform(0.01,3.0),ra.uniform(0.01,3.0),ra.uniform(0.01,3.0),ra.uniform(0.01,3.0)])
			    
			    Ini = np.array([ra.uniform(0.01,6.0),ra.uniform(0.01,9.0),ra.uniform(0.01,5.0),ra.uniform(0.01,9.0)])

			    Iter1 = 30
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
							    f_series.write('%.5f %.5f %.5f %.5f %.5f %.5f\n'% (n, p_b, Ini[0], Ini[1], Ini[2], Ini[3]))
			    #if jj == Iter-1: print (Ind)
			    
	print (Ind_1)
	


#if __name__ == '__main__': N_Proces = 30; Pool(N_Proces).map(FUN, zip(np.linspace(1.0, 2.4, POINTS), range(POINTS)))
#print FUN((1.5,1000,0))







# Merge the files
def mergeFiles(input):
    for index in range(POINTS):
        DATA = [];
        for Ind in range(int(POINTS*times_n)):
            dataa = np.genfromtxt(PATH+'/junk/il4_{}_{}.dat'.format(index, Ind))
            if len(dataa) > 0: DATA.append(dataa)


        with open(os.path.join(PATH+'/data_2p/il4_n_b_{}.dat'.format(index)), 'w') as f_series:
            for data, LE in zip(DATA, range(len(DATA))):
                if len(data) > 0:
                    Data = np.unique(data,axis=0)  # After removing all same roots/data
                    if len(np.shape(data)) > 1:   # Will check either number of different roors are there or not
                        for da in Data:
                            f_series.write('%.5f %.5f %.5f %.5f %.5f\n'% (da[0],da[1],da[2],da[3],da[4]))		
                    if len(np.shape(data)) == 1: # Will check if there is only one root
                        f_series.write('%.5f %.5f %.5f %.5f %.5f\n'% (data[0],data[1],data[2],data[3],da[4]))		
                else: # If no root found then again run the program
                        print (LE)	

        print (index)
        
#print (mergeFiles(0))






def bistable(input):

    final_bistable = []
    final_transition = []
    
    for index in range(POINTS):
        data = np.genfromtxt(PATH+'/data_2p/il4_n_b_{}.dat'.format(index))
        
        variable = []
        transition = []
        
        # To get start point of bistability
        compare = dict()
        for data_c, index_0 in zip(data, range(len(data))):
            if data_c[1] in compare:
                if compare[data_c[1]] != np.around(data_c[2], decimals=1):
                    variable.extend([data_c[0], data_c[1]])
                    print (data_c[0], data_c[1]); break
            if data_c[1] not in compare:
                compare[data_c[1]] = np.around(data_c[2], decimals=1)
                
        # To get end point of bistability         
        compare = dict()
        for data_c, index_0 in zip(data[::-1], range(len(data))[::-1]):
            if data_c[1] in compare:
                if compare[data_c[1]] != np.around(data_c[3], decimals=1):
                    variable.append(data_c[1])
                    print (data_c[0], data_c[1]); break
            if data_c[1] not in compare:
                compare[data_c[1]] = np.around(data_c[3], decimals=1)
                
        # To get interaction point        
        for data_c, index_0 in zip(data[3:], range(len(data[3:]))):
            if data_c[2] < data_c[3]: 
                transition.extend([data_c[0], data_c[1]])
                break
                
        if len(variable) == 3: final_bistable.append(variable)
        final_transition.append(transition)

    np.savetxt(PATH+'/data_2p/il4_n_b_bistable.dat', final_bistable, fmt='%0.5f')
    np.savetxt(PATH+'/data_2p/il4_n_b_transition.dat', final_transition, fmt='%0.5f')    
    
print (bistable(0))
#data = np.genfromtxt(PATH+'/data_2p/il4_n_b_29.dat'); plt.plot(data[:,1], data[:,2], '.r', data[:,1], data[:,3], '.b', markersize=1.0); plt.show()


data = np.genfromtxt(PATH+'/data_2p/il4_n_b_bistable.dat'); 
plt.plot(data[:,0], data[:,1], '.r', data[:,0], data[:,2], '.r', markersize=2.0)

data = np.genfromtxt(PATH+'/data_2p/il4_n_b_transition.dat'); 
plt.plot(data[:,0], data[:,1], '.b', markersize=2.0);
plt.tight_layout(); plt.show()


end_time = datetime.now()
print ('MAIN PROGRAM IS COMPLETED||Duration||H:M:S||{}'.format(end_time - start_time), '\n')
####################################################################################################

