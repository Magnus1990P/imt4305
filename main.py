#!/usr/bin/env python

from 	cv2						import *
from numpy 					import *
import sys
import scipy 				as sp
import scipy.misc 	as spMisc
import scipy.signal	as spSignal
import matplotlib.pyplot as plt

spHeight	= 1
spWidth		= 1
spPos			= 1		

filters		= ["Low-Pass","High-Pass",]
genIMG		= {}
metrics		= {}

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
	


def genHPFilter(shape=(6,6),sigma=0.5):
	m,n 	= [(ss-1.)/2.0 for ss in shape]
	y,x 	= mgrid[-m:m+1, -n:n+1]
	h 		= exp( -(x*x + y*y) / (2.0*(sigma**2)) )

	print h	
	plt.plot(h)
	plt.imshow(h)
	

	



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
	pass
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


y = len(oIMG)
x = len(oIMG[1])
#genHPFilter( )

fltr = [ [1./9,]*3,	]*3
print fltr
fImg = spSignal.convolve2d(abs(genIMG["Shifted FFT"][1]), fltr, mode="same")
plt.imshow(fImg)
#applyFilter( oIMG, fltr )

#showImage()

#Compare images
#compare("Shifted FFT")

#compare("Reconstruced (Mag * phase)")
#applyFilter()
#PLOT IMAGE


