#!/usr/bin/env python
################################################################################
##
##	
##
##
##
################################################################################
################################################################################
################################################################################
################################################################################
from		cv2								import *
from		numpy 						import *
import	sys
import	scipy 						as sp
import	scipy.misc 				as spMisc
import	scipy.signal			as spSignal
import	matplotlib.pyplot as plt





fIMG	= None
oIMG	= None
OUTPUT = False

spHeight	= 1
spWidth		= 1
spPos			= 1		

genIMG		= {}
metrics		= {}
filters		= sorted(["LP: Ideal", "LP: Buttwerworth", "LP: Gaussian", 
										"HP: Ideal", "HP: Buttwerworth", "HP: Gaussian"])




################################################################################
#
# Function which output logging when desired
#
#	@param TEXT:	String containing the output string
################################################################################
def console_log( TEXT=" " ):
	if OUTPUT:
		print TEXT

################################################################################
#
# Calculates the histogram of the image
#
#	@param img:	MxN size matrix
#	@return hgram:	1x256 matrix containg counters for each graylevel value
################################################################################
def calcHistogram( img ):
	hgram = [0,]*256 																		#Generate 255 long array
	console_log("\tCalculating the histogram of image")
	for y in img:
		for x in y:
			hgram[int(x)] = hgram[int(x)]+1									#incrementing cell counter
	return hgram	


################################################################################
#
# 
#
#	@param :	
#
#	@return :		
#
################################################################################
def rmse(org, diff):																	#Root Mean Squared Error
	console_log("RMSE")
	RMSE	= sqrt( ( (diff - org)**2 ).mean() )					#Calculating RMSE
	return RMSE




################################################################################
##
## This code in its entirety is collected and converted from the code published on
## the University of Regina, Department of Computer Science's websites below
## http://www.cs.uregina.ca/Links/class-info/425-nova/Lab5/M-Functions/lpfilter.m
## http://www.cs.uregina.ca/Links/class-info/425-nova/Lab5/M-Functions/hpfilter.m
##
################################################################################
#
#Calculate DFTUV of MxN matrix
# Generate two padded meshgrids
#
#	@param M:			Width
#	@param N:			Height
# @return V:		Meshgrid of size MxN
#	@return U:		Meshgrid of size MxN
################################################################################
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


################################################################################
##
## This code in its entirety is collected and converted from the code published on
## the University of Regina, Department of Computer Science's websites below
## http://www.cs.uregina.ca/Links/class-info/425-nova/Lab5/M-Functions/lpfilter.m
## http://www.cs.uregina.ca/Links/class-info/425-nova/Lab5/M-Functions/hpfilter.m
##
################################################################################
#
# Generate low pass filters of MxN size.
#
#	@param type:	Type of filter to generate. ideal, butterworth or gaussian
# @param M:			Height of matrix
# @param N:			Width of matrix
# @param D0:		Cut-off frequency.  Positive float value
# @param n:			multiplication value for butterworth, default 1.0 if not set
#	@return H:		matrix of MxN
################################################################################
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
		console_log("Unknown filter type, quitting: %s" % t)
		sys.exit(0)
	return H


################################################################################
##
## This code in its entirety is collected and converted from the code published on
## the University of Regina, Department of Computer Science's websites below
## http://www.cs.uregina.ca/Links/class-info/425-nova/Lab5/M-Functions/lpfilter.m
## http://www.cs.uregina.ca/Links/class-info/425-nova/Lab5/M-Functions/hpfilter.m
##
################################################################################
#
# Generate low pass filters of MxN size.
#
#	@param type:	Type of filter to generate. ideal, butterworth or gaussian
# @param M:			Height of matrix
# @param N:			Width of matrix
# @param D0:		Cut-off frequency.  Positive float value
# @param n:			multiplication value for butterworth, default 1.0 if not set
#	@return H:		matrix of MxN
################################################################################
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


################################################################################
##
## This code in its entirety is collected and converted from the code published on
## the University of Regina, Department of Computer Science's websites below
## http://www.cs.uregina.ca/Links/class-info/425-nova/Lab5/M-Functions/lpfilter.m
## http://www.cs.uregina.ca/Links/class-info/425-nova/Lab5/M-Functions/hpfilter.m
##
################################################################################
#
# Generates a high pass filters of MxN size.
#
#	@param type:	Type of filter to generate. ideal, butterworth or gaussian
# @param M:			Height of matrix
# @param N:			Width of matrix
# @param D0:		Cut-off frequency.  Positive float value
# @param n:			multiplication value for butterworth, default 1.0 if not set
#	@return H:		matrix of MxN, where the content of each cell is (1-x)
################################################################################
def genHPFilter(t, M, N, D0=100.0, n=1.0):
	Hlp = genLPFilter(t, M, N, D0, n)
	Hhp = (1 - Hlp)
	return Hhp







################################################################################
#
# Performa  set of actions on the image as defined by an array
#
#	@param oimg:	Matrix representation of image, of MxN size
#	@param arr:	Array of indexes for action to perform
#	@return aImg: Matrix representation of image, of MxN size with the actions
#	performed		
################################################################################
def imgAction(oimg, arr):
	img = oimg[:]					#REplicate image into new matrix

	console_log("Performing the following actions on the provided image.")
	for i in arr:
		#ABS
		if i == 0:
			console_log("\tTaking the absolute values of each pixel")
			img = abs(img)
		#IFFT2
		elif i == 1:
			console_log("\tCalculating the inverse the fourier transform")
			img = fft.ifft2( img )
		#IFFT
		elif i == 2:
			console_log("\tCalculating the inverse the fourier shift")
			img = fft.ifftshift( img )
		#Log representaiton
		elif i == 3:
			console_log("\tCalculate the logarithmic representation of image")
			img = log( 1 + img )

		#FFT2
		elif i == 10:
			console_log("\tCalculate the 2D Fourier transform of the image")
			img = fft.fft2( img )
		#FFTSHift
		elif i == 11:
			console_log("\tCalculate the 2D fourier shift of the image")
			img = fft.fftshift( img )
		#Real
		elif i == 12:
			console_log("\tCalculate the real part of the image")
			img = real( img )
		#Imaginary
		elif i == 13:
			console_log("\tCalculate the imaginary part of the image")
			img = imag( img )
		#Phase
		elif i == 14:
			console_log("\tCalculate the phase of the image")
			img = exp(1j * angle(img))
		#Magnitude
		elif i == 15:
			console_log("\tCalculating the magnitude of FFT2")
			img = abs(img)

	return img

################################################################################
#
# Generates a set of basic images from the originally loaded image
#
################################################################################
def splitThatImage( ):
	global genIMG
	genIMG.update({'FFT':	
								[[1,0],		imgAction(oIMG[:],[10])] })
	genIMG.update({'Shifted FFT':
								[[2,1,0],	imgAction(oIMG[:],[10,11])] })
	genIMG.update({'Real part of FFTSHIFT':	
								[[2,1,0],	imgAction(oIMG[:], [10,11,12])]	})
	genIMG.update({'Imaginary part of FFTSHIFT':		
								[[2,1,0],	imgAction(oIMG[:], [10,11,13])]	})
	genIMG.update({'Phase of FFTSHIFT':
								[[2,1,0],	imgAction(oIMG[:], [10,11,14,])]	})
	genIMG.update({'Magnitude of FFTSHIFT':
								[[3,0],	imgAction(oIMG[:], [10,11,15])]	})

	genIMG.update({'Reconstructed (Mag * phase)': [[2,1,0],
		(imgAction(oIMG[:],[10,11,15])[:] * imgAction(oIMG[:], [10,11,14])[:])]})


################################################################################
#
# Set the size of the imageboard when displaying images
#
#	@param arr: 2 cell list containing numbers for choosing the size in Y,X	
################################################################################
def setSPlotSize(arr):
	global spHeight, spWidth,spPos
	console_log("Max number of subplots was changed from %d to %d" % (spHeight*spWidth, arr[0]*arr[1]))
	spHeight	= arr[0]
	spWidth 	= arr[1]
	spPos 		= 1



################################################################################
#
# Plot images on the imageboard of Y,X size
#
#	@param pos:	The position it should have in the  imaage board, currently spPos
#	@param arr:	The reversal process for the image
#	@param img:	image represented as an MxN matrix
#	@param title:	title of image
################################################################################
def sPlot(pos, arr, img, title):
	global spPos
	if pos > spHeight*spWidth:
		console_log("Position is out of bounds, adjust and replot")
		return None
		
	console_log("Subplotting (" + str(title) + ") @ position "+ str(pos) + "," + str(pos))
	plt.subplot(spHeight,spWidth,pos)
	plt.title( title )

	if 20 in arr:		#if its a histogram or multiple histograms in same subplot
		console_log("Histogram plotting")
		leg = []
		for i in range(1,len(img)):
			plt.plot(range(0,256), img[i])
			leg.append("Image #"+str(i))
		plt.legend(leg)
		plt.axis([0, 256, img[0][0], img[0][1]])
	
	else:						#If image
		plt.imshow( imgAction(img, arr), cmap="gray"  )

	if spPos < (spHeight*spWidth):
		spPos		= spPos + 1
	else:
		print "Position variable has reached its max(%i)." % spPos
	

################################################################################
#
# Choose images from a set of generated images
#
#	@return tList: List of titles found in the dictionary genIMG
################################################################################
def chooseImages():
	i = 0;																				#
	titles = sorted(genIMG.keys())								#Grab list of titles, sorted
	print "Images in processing list (%d)" % len(titles)	#
	for title in titles:													#Print the titles to screen
		print "%3d:\t%s" % (i, title)								#Print the titles
		i = i + 1																		#

	l = raw_input("Selected images (1,2,3):\t")		#Select images in genIMGS
	l = map(int, l.strip("\t\n\0").split(","))		#Convert list to integers

	tList = []
	for ind in l:
		tList.append( titles[ind] )

	return tList
	


################################################################################
#
# Displays a set of chosen images
#
################################################################################
def showImages():
	l = raw_input("Set imageboard layout (height,width):\t")			#Grab layout X,Y
	l = l.strip("()\t\n\0").split(",")							#
	setSPlotSize( map(int, l) );										#Set display layout

	titles = chooseImages()													#Choose a set of images

	for T in titles:																#Plot all images in the list
		sPlot(spPos, genIMG[T][0], genIMG[T][1], T)		#Plot images 
	plt.show()																			#Display the imageboard



################################################################################
#
# Generate histogram metrics
#
#	@param h1:	Histogram of original image
#	@param h2:	Histogram of altered image
################################################################################
def histComp(h1,h2):
	global metrics
	DLman = DLeuc = DLcos = DLmat = 0
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
	

################################################################################
#
# Compare altered image agains the original image
#
#	@param title: Title of altered image to compare	
################################################################################
def compare( title ):
	global metrics
	setSPlotSize([2,2]);
	dImgTitle = title
	dImgArr		= genIMG[title][0][:]
	dImgImg		= genIMG[title][1][:]

	sPlot(spPos, [0], oIMG, "Original image B/W")
	sPlot(spPos, [0,1], dImgImg, dImgTitle)	

	histOrg = histogram( oIMG )
	histFil	= histogram( imgAction(dImgImg, dImgArr) )
	histDif	= histFil[:]
	metrics.update({"RMSE": rmse(imgAction(dImgImg, dImgArr), oIMG)})

	for i in range(0,len(histOrg)):
		histDif[i] = abs(histOrg[i] - histDif[i])

	#Plot histograms
	sPlot(spPos, [20], [[0, amax(histOrg)+500 ], histOrg, histFil], 
				"Histogram - both images")
	sPlot(spPos, [20], [[0, amax(histDif)+100 ], histDif],
				"Histogram - diff")

	plt.show()
	histComp(histOrg,histFil)	
	for i in metrics.keys():
		console_log("%50s: %20s" %(i,str(metrics[i])))
	metrics.update({"orgHist":histOrg})	
	metrics.update({"filHist":histFil})	
	metrics.update({"difHist":histDif})



################################################################################
#
# Function for choosing which filter to apply to an image
#
#	@return filters[choice]:	Returns the title of the chosen filter to apply
################################################################################
def chooseFilter( ):
	console_log("Available filters: ")
	i = 0
	for f in filters:
		console_log("%2d: %s" % (i, f))
		i += 1
	console_log( )
	choice = int( raw_input("Choose filter: ").rstrip() )
	return filters[choice]


################################################################################
#
# Applies a filter which is chosen to the original image
#
################################################################################
def applyFilter( D0=100 ):
	global genIMG
#	fltr = chooseFilter()								#Choose filter to apply
	fltr = "LP: Gaussian"
	fltr = "LP: Buttwerworth"
	console_log("You chose the %s, please input data or use its default" % fltr)
	
	iFFTS = genIMG["Shifted FFT"][1]		#Grab shifted FFT2 of orignal image
	M,N 	= iFFTS.shape									#Get the dimensions
	t 		= "ideal"											#Default filter
	H 		= None												#Resulting filter
	n 		= 1.0													#btw-multiplier

	#print "Cut-off frequency as single integer or float greater than 0."
	#D0	= float(raw_input("D0:\t"))		#grab Cut-off value

	if fltr == "LP: Buttwerworth":
		t = "btw"
		print "Butterwort multiplier value. Float value <= 1.0"
		n	= float(raw_input("n:\t"))
	elif fltr == "LP: Gaussian":
		t = "gaussian"
	elif fltr == "HP: Buttwerworth":
		print "Butterwort multiplier value. Float value equal to or greater than 1.0"
		n	= float(raw_input("n:\t"))
	elif fltr == "HP: Gaussian":
		t = "gaussian"
	if fltr[0:2] == "LP":
		H = genLPFilter(t,M,N,D0,n)
	elif fltr[0:2] == "HP":
		H = genHPFilter(t,M,N,D0,n)
	else:
		print "Unknown Filter, quitting"
		sys.exit(0)

	H 		= imgAction(H,[2])
	fLP		= (H*iFFTS)					#Create filter (FFT)
	m 		= (H+(1*fLP))

	k = 10
	setSPlotSize([2,2])														#Set image layout

	sPlot(spPos, [3,0],			(iFFTS+(k*m)),	"")			#image + mask
	sPlot(spPos, [2,1,0],		(iFFTS+(k*m)),	"")			#image + mask


	H2    = (1-H)
	fLP2	= (H2*iFFTS)					#Create filter (FFT)
	m2 		= (H2+(1*fLP2))

	sPlot(spPos, [3,0],		(iFFTS+(k*m2)),	"")			#image + mask
	sPlot(spPos, [2,1,0],		(iFFTS+(k*m2)),	"")			#image + mask


	plt.show()																					#Show images
	
	








################################################################################
#
# Check if parameters was provided
#
################################################################################
if len(sys.argv) <= 1:
	print "not valid argument"
	sys.exit(0)
if len(sys.argv) > 2 and sys.argv[-1]=="TRUE":
	OUTPUT=True


################################################################################
#
# Do default stuff.  Read filename, load image, split the image into different
#	versions(FFT, FFT Shifted, Phase, Imaginary, real and magnitude)
#
################################################################################
fIMG	= sys.argv[1]
oIMG	= imread( fIMG, 0 )
genIMG.update({"Original image B/W":[[0],oIMG]})
splitThatImage()

applyFilter(25)






################################################################################
#
# 
#
#	@param :	
#
#	@return :		
#
################################################################################
