"""
Smoothing and Predicting given a Dyanmic Bayesian Network 
Consider a time series network. Assume that the state
transition probabilities are given by P(Xt+1 = T|Xt = T) = 0.75 and P(Xt+1 = T|Xt = F) = 0.25 for
t = {0, 1, 2, 3, 4, 5}, and by P(Xt+1 = T|Xt = T) = 0.81 and P(Xt+1 = T|Xt = F) = 0.16 for all t > 5.
The evidence probabilities are given by P(Et = T|Xt = T) = 0.71 and P(Et = T|Xt = F) = 0.01. Let
E1:6 = {T, F, T, F, T, F}. Let the prior P(X0 = T) = 0.2.

Program to estimate the predictive probability

lim
k→∞    P(X6+k = T|E1:6)
		
Program to estimate the smoothed probability P(X3 = T|E1:6)
"""

from numpy import matrix
import numpy as np
from sklearn.preprocessing import normalize  #Ran this on a CSE lab machine and it worked - just used to normalize by row or column

Ot = matrix( [[0.71,0],[0,0.01]])   #emission/observation matrix for true case
Of = matrix( [[0.29,0],[0,0.99]])   #emission/observation matrix for false case

Tls5 = matrix( [[0.75,0.25],[0.25,0.75]])  #transition matrix for case where t <= 5
Tgr5 = matrix( [[0.81,0.19],[0.16,0.84]])  #transition matrix for case where t > 5

def fwd_bkw(evidence, prior):
	#evidence is a list of evidence values for steps 1,..,t
	#prior is P(x0)
	fv = [None]*(len(evidence)+1) 		#a vector of forward messages
	b = matrix([[1],[1]])	  		#a backward messages initialized to 1
	sv = [None]*len(evidence)  		#a vector of smoothed estimates for steps 1,..,t
	
	fv[0] = prior                           #set initial forward message to the prior
	for k in range(1,len(evidence)+1):        
		fv[k] = fNorm(forward(fv[k-1],evidence[k-1],k))  #need evidence of k-1 because there is no evidence for time t = 0 so evidence for time t = 1 is stored in 0th position of evidence list
	for k in range(len(evidence),0,-1):			#0 because range is not inclusive of last number
		sv[k-1] = merge(fv[k],b)                        #Store smoothed term
		b = bNorm(backward(b, evidence[k-1],k))
		
	return sv
	
def forward(f_mes, ev, t):

	if ev == 'T':
		if t > 5:
			return f_mes*Tgr5*Ot
		else:
			return f_mes*Tls5*Ot		
	elif ev == 'F': 							
		if t > 5:		
			return f_mes*Tgr5*Of

		else:
			return f_mes*Tls5*Of	
			
def backward(b_mes, ev, t):
	if ev == 'T':
		if t > 5:
			return Tgr5*Ot*b_mes
		else:
			return Tls5*Ot*b_mes		
	elif ev == 'F': 							
		if t > 5:
			return Tgr5*Of*b_mes
		else:
			return Tls5*Of*b_mes		
			
def fNorm(mat):
	return normalize(mat, axis = 1, norm='l1')   #just nromalizes
					
def bNorm(mat):
	return normalize(mat, axis = 0, norm='l1')	#just normalizes
	
def merge(f, b):
	return(fNorm(np.asmatrix(np.array(f)*np.array(b.T))))  #convert to array to do point-wise product but convert back to matrix to normalize	
	
def prediction():
	prior = matrix([0.2,0.8])
	evidence = ['T','F','T','F','T','F']
	fv = [None]*(len(evidence)+1) 		#a vector of forward messages
	
	fv[0] = prior
	for k in range(1,len(evidence)+1):        
		fv[k] = fNorm(forward(fv[k-1],evidence[k-1],k))
	lastFv = fv[6]
	
	oldPred = lastFv
	j=0
	for i in range(0,5000):
		newPred = oldPred*Tgr5
		if np.array_equal(np.array(oldPred),np.array(newPred)):
			break		
		oldPred = newPred
	print("Problem 4C (lim k-> inf P(X6+k = T|E1:6) took " + str(i) + " steps to converge to " +str(newPred.item(0,0)))	

def smoothing():
	p = matrix([0.2,0.8])
	e = ['T','F','T','F','T','F']
	print("Problem 4D (P(X3 = T|E1:6)): "+ str(fwd_bkw(e,p)[2].item(0,0)))


		
prediction()
smoothing()

	
					
