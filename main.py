#!/usr/bin/env python

from		cv2								import *
from		numpy 						import *
import	sys
import	scipy 						as sp
import	scipy.misc 				as spMisc
import	scipy.signal			as spSignal
import	matplotlib.pyplot as plt
from		uc_regina_filter	import	*

spHeight	= 1
spWidth		= 1
spPos			= 1		

filters		= ["Low-Pass","High-Pass",]
genIMG		= {}
metrics		= {}


#This code in its entirety is collected and converted from the code published on
#the University of Regina, Department of Computer Science's websites below
#http://www.cs.uregina.ca/Links/class-info/425-nova/Lab5/M-Functions/lpfilter.m
#http://www.cs.uregina.ca/Links/class-info/425-nova/Lab5/M-Functions/hpfilter.m
#
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


#This code in its entirety is collected and converted from the code published on
#the University of Regina, Department of Computer Science's websites below
#http://www.cs.uregina.ca/Links/class-info/425-nova/Lab5/M-Functions/lpfilter.m
#http://www.cs.uregina.ca/Links/class-info/425-nova/Lab5/M-Functions/hpfilter.m
#
# Generate low pass filters of MxN size.
#
#	@param type:	Type of filter to generate. ideal, butterworth or gaussian
# @param M:			Height of matrix
# @param N:			Width of matrix
# @param D0:		Cut-off frequency.  Positive float value
# @param n:			multiplication value for butterworth, default 1.0 if not set
#
#	@return H:		matrix of MxN
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
		sys.exit(0)
	return H


#This code in its entirety is collected and converted from the code published on
#the University of Regina, Department of Computer Science's websites below
#http://www.cs.uregina.ca/Links/class-info/425-nova/Lab5/M-Functions/lpfilter.m
#http://www.cs.uregina.ca/Links/class-info/425-nova/Lab5/M-Functions/hpfilter.m
#
# Generates a high pass filters of MxN size.
#
#	@param type:	Type of filter to generate. ideal, butterworth or gaussian
# @param M:			Height of matrix
# @param N:			Width of matrix
# @param D0:		Cut-off frequency.  Positive float value
# @param n:			multiplication value for butterworth, default 1.0 if not set
#
#	@return H:		matrix of MxN, where the content of each cell is (1-x)
def genHPFilter(t, M, N, D0=100.0, n=1.0):
	Hlp = genLPFilter(t, M, N, D0, n)
	Hhp = (1 - Hlp)
	return Hhp


def setSPlotSize(arr):
	global spHeight, spWidth,spPos
	print "Max number of subplots was changed from %d to %d" % (spHeight*spWidth, arr[0]*arr[1])
	spHeight	= arr[0]
	spWidth 	= arr[1]
	spPos 		= 1


def revertImg(img, arr):
	print "reverting image"
	for i in arr:
		if i == 1:		#Fourier Transformed FFT2
			print "Inverse the fourier transform"
			img = fft.ifft2( img )
		elif i == 2:	#Shifted fourier transform
			print "Inverse the fourier shift"
			img = fft.ifftshift( img )
		elif i == 3:	#Requiers log to be displayed
			print "Generate logarithmic representation of image"
			img = log( 1 + img )
		elif i == 4:	#TBD
			a=0

	if 0 in arr:		#if needs abs to display
		img = abs( img )
	print 
	return img

def splitThatImage( img ):
	global genIMG
	#Fourier transform of 2D image
	print "Generating Fourier transforms of images"
	F = fft.fftshift( fft.fft2( img ) )

	genIMG.update({'Shifted FFT':											[[0,1],		F[:]]				})
	genIMG.update({'Real part of shifted FFT':				[[0,1],		real(F[:])]	})
	genIMG.update({'Imaginary part of shifted FFT':		[[0,1],		imag(F[:])]	})
	genIMG.update({'Magnitude of shifted FFT':				[[0,3],		abs(F[:])]	})
	genIMG.update({'Phase of shifted FFT':						[[0,1],		exp( 1j * angle(F[:]))]})
	genIMG.update({'Reconstruced (Mag * phase)':			[[0,1],		(abs(F[:]) * exp(1j * angle(F[:])))]})
	
	
def sPlot(pos, arr, img, title):
	global spPos
	if pos > spHeight*spWidth:
		print "Position is out of bounds, adjust and replot"
		return None
		
	print "Subplotting (" + str(title) + ") @ position "+ str(pos) + "," + str(pos)
	plt.subplot(spHeight,spWidth,pos)
	plt.title( title )

	if 20 in arr:		#if its a histogram or multiple histograms in same subplot
		print "Histogram plotting"
		leg = []
		for i in range(1,len(img)):
			plt.plot(range(0,256), img[i])
			leg.append("Image #"+str(i))
		plt.legend(leg)
		plt.axis([0, 256, img[0][0], img[0][1]])
	
	else:						#If image
		plt.imshow( revertImg(img,arr), cmap="gray"  )

	if spPos < (spHeight*spWidth):
		spPos		= spPos + 1
	else:
		print "position variable is reached maximal value of %i, revise size if needed" % spPos
	



def histogram( img ):
	hgram = [0,]*256
	for y in img:
		for x in y:
			hgram[int(x)] = hgram[int(x)]+1
	return hgram	


def rmse(org, diff):
	print "RMSE"
	RMSE	= sqrt( ( (diff - org)**2 ).mean() )
	return RMSE



def showImage():
	global spPos;
	print "Images in processing list"
	i = 0;
	titles = sorted(genIMG.keys())
	for title in titles:
		print "%3d:\t%s" % (i, title)
		i = i + 1
	print ""

	l = raw_input("Layout (height,width):")
	l = l.strip("()\t\n\0").split(",")
	setSPlotSize( map(int, l) );

	l = raw_input("Selected images (1,2,3): ")
	l = map(int, l.strip("\t\n\0").split(","))
	
	spPos = 1;
	for i in l:
		sPlot( spPos, genIMG[titles[i]][0], genIMG[titles[i]][1], titles[i]  )
	plt.show()

def histComp(h1,h2):
	global metrics
	DLman=0
	DLeuc=0
	DLcos=0
	DLmat=0
	for i in range(0,256):
		DLman = DLman + abs(h1[i] - h2[i])
		DLeuc = DLeuc + (abs(h1[i] - h2[i])**2)
		DLcos = DLcos + (h1[i]*h2[i])
		DLmat = DLmat + abs(h1[i]-h2[i])
	DLeuc = sqrt( DLeuc )
	DLcos = 1 - DLcos
	metrics.update({"Histogram - Dist Manhattan":DLman})
	metrics.update({"Histogram - Dist Eucledian":DLeuc})
	metrics.update({"Histogram - Dist Cosine":DLcos})
	metrics.update({"Histogram - Dist Match":DLmat})
	

def compare( title ):
	global metrics
	setSPlotSize([2,2]);
	dImgTitle = title
	dImgArr		= genIMG[title][0][:]
	dImgImg		= genIMG[title][1][:]

	sPlot(spPos, [0], oIMG, "Original image B/W")
	sPlot(spPos, [0,1], dImgImg, dImgTitle)	

	histOrg = histogram( oIMG )
	histFil	= histogram( revertImg(dImgImg, dImgArr) )
	histDif	= histFil[:]
	metrics.update({"RMSE": rmse(revertImg(dImgImg, dImgArr), oIMG)})

	for i in range(0,len(histOrg)):
		histDif[i] = abs(histOrg[i] - histDif[i])

	sPlot(spPos, [20], [[0,amax(histOrg)+500],histOrg,histFil], "Histogram - both images")
	sPlot(spPos, [20], [[0,amax(histDif)+100], histDif], "Histogram - diff")

	plt.show()
	histComp(histOrg,histFil)	
	for i in metrics.keys():
		print "%50s: %20s" %(i,str(metrics[i]))
	metrics.update({"orgHist":histOrg})	
	metrics.update({"filHist":histFil})	
	metrics.update({"difHist":histDif})


def chooseFilter( ):
	print "Available filters: "
	i = 0
	fList = sorted(filters.keys())
	for f in fList:
		print "%3d: %s" % (i, f)
	print
	choice = int( raw_input("Choose filter: ").rstrip() )
	print choice

def applyFilter( img, fltr ):
	passy
	#f = chooseFilter()


#INPUT CHECK
if len(sys.argv) <= 1:
	print "not valid argument"
	sys.exit(0)


#READ AND SPLIT THE IMAGE
fIMG	= sys.argv[1]
oIMG	= imread( fIMG, 0 )
genIMG.update({"Original image B/W":[[0],oIMG]})
splitThatImage( oIMG );


M = len( oIMG )
N = len( oIMG[1] )

H = genLPFilter( "ideal", 5, 5, 2.5, 1.0)
print H
H = genLPFilter( "btw", 	5, 5, 2.5, 1.0)
print H
H = genLPFilter( "gaussian", 	5, 5, 2.5, 1.0)
print H
print
print
H = genHPFilter( "ideal", 5, 5, 2.5, 1.0)
print H
H = genHPFilter( "btw", 	5, 5, 2.5, 1.0)
print H
H = genHPFilter( "gaussian", 	5, 5, 2.5, 1.0)
print H
print
print









#h = genIMG["Shifted FFT"][1]*H

#plt.imshow( abs(log(1+h)), cmap="gray"  )
#plt.show()



#showImage()

#Compare images
#compare("Shifted FFT")

#compare("Reconstruced (Mag * phase)")
#applyFilter()
#PLOT IMAGE


