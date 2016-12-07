#!/usr/bin/env python
#This code in its entirety is collected and converted from the code published on
#the University of Regina, Department of Computer Science's websites below
#
#http://www.cs.uregina.ca/Links/class-info/425-nova/Lab5/M-Functions/lpfilter.m
#http://www.cs.uregina.ca/Links/class-info/425-nova/Lab5/M-Functions/hpfilter.m
#http://www.cs.uregina.ca/Links/class-info/425-nova/Lab5/M-Functions/dfutv.m


#Calculate DFTUV of MxN matrix
# Generate two padded meshgrids
#
#	@param M:			Width
#	@param N:			Height
#
# @return V:		Meshgrid of size MxN
#	@return U:		Meshgrid of size MxN
def dftuv(M, N):
	idx	= idy	= 0
	u = range(0,M)
	v = range(0,N)
	for idx in range(0, len(u)):			#Convert all elements greater than M/2
		if u[idx] > (M/2):
			u[idx] = u[idx] - M
	for idy in range(0, len(v)):			#Convert all elements greater than N/2
		if v[idy] > (N/2):
			v[idy] = v[idy] - N
	V,U = meshgrid(v, u)							#Generate meshgrid
	return [V,U]											#Return meshgrid


# Generate low pass filters of MxN size.
#
#	@param type:	Type of filter to generate. ideal, butterworth or gaussian
# @param M:			Height of matrix
# @param N:			Width of matrix
# @param D0:		Cut-off frequency.  Positive float value
# @param n:			multiplication value for butterworth, default 1.0 if not set
#
#	@return H:		matrix of MxN MxNe
def genLPFilter(t, M, N, D0=100, n=1.0):
	U,V = dftuv( M, N )										# Compute padded meshgrid
	D = sqrt( power(U,2) + power(V,2) )		# square root of array
	H = array(D)													# Convert to array
	if t == "ideal":
		for y in range(0, len(H)):
			for x in range(0, len(H[y])):
				H[y][x] = 1 if H[y][x]<=D0 else 0

	elif t == "btw":
		H = 1.0 / (1 + power((H/D0), (2*n)) )
	
	elif t == "gaussian":
		H = exp( (-1*power(D,2)) / (2*(D0**2)) )
	
	else:
		print "Unknown filter type, quitting: %s" % t
		exit(0)
	return H


