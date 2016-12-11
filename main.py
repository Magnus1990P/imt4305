#!/usr/bin/env python
################################################################################
##
##	@author:	Magnus1990P
##	@Name:		Magnus Øverbø
##	@date:		2016.12.11
##	
##	This software takes an arbitrary input image and reads it as a grayscale
##	image.  Then it gives you the option to perform an arbitrary number of
##	operations on the image. (Given that they are listed and programmed.
##
##	Default it has three methods for sharpening, three methods for blur and
##	three methods for reducing noise.
##
##	It performs comparison by generating the histogram and RMSE of two images
##	then storing the resulting metrics for comparison later on.
##
##	The program also allows for saving or discarding images when applying filter
##	and automatically stores images upon exiting the software.
##	It has hardoced names and will overwrite any existing files in the outdir.
##
################################################################################
################################################################################
##
##	Import libraries
##
################################################################################
################################################################################
from		cv2								import *
from		numpy 						import *
import	sys
import	scipy 						as sp
import	scipy.misc 				as spMisc
import	scipy.signal			as spSignal
import	matplotlib.pyplot as plt



################################################################################
################################################################################
##
##	Global variables
##
################################################################################
################################################################################
fIMG	= None
oIMG	= None
OUTPUT = False

spHeight	= 1
spWidth		= 1
spPos			= 1		

genIMG		= {}
filters		= sorted(["LP: Ideal", "LP: Buttwerworth", "LP: Gaussian", 
										"Noise: Line rotate",
										"Noise: Local spots",
										"Noise: Horizontal-Vertical",
										"HP: Ideal", "HP: Buttwerworth", "HP: Gaussian"])


################################################################################
################################################################################
##
##  FUNCTIONS FROM UC REGINA
##
################################################################################
################################################################################
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
################################################################################
##
##  CUSTOM FUNCTIONS WRITTEN FOR IMPLEMENTING THE FUNCTIONALITY IN THIS PROGRAM
##
################################################################################
################################################################################
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
		if T in ["Shifted FFT", "FFT"]:
			sPlot(spPos, [3,0], genIMG[T][1], T)		#Plot images 
		else:
			sPlot(spPos, genIMG[T][0], genIMG[T][1], T)		#Plot images 
	plt.show()																			#Display the imageboard


################################################################################
#
# Generate histogram metrics
#
#	@param h1:	Histogram of original image
#	@param h2:	Histogram of altered image
#	@param metrics:	Dictionary for metrics
################################################################################
def histComp(h1,h2, metrics):
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
# Calculates the histogram of the image
#
#	@param img:	MxN size matrix
#	@return hgram:	1x256 matrix containg counters for each graylevel value
################################################################################
def calcHistogram( img ):
	hgram = [0,]*256 																			#Generate 255 long array
	console_log("\tCalculating the histogram of image")
	for y in img:
		for x in y:
			x = int(x) % len(hgram) 									#Grab value betwwon 0-255
			hgram[x] = hgram[x]+1											#incrementing cell counter
	return hgram																	#Return histogram


################################################################################
#
#	Root Mean Squared Error 
#
#	@param 	org:	original image histogram
#	@param	diff:	the altered image histogram
#	@return RMSE:	The value of RMSE	
################################################################################
def rmse(org, diff):																	#Root Mean Squared Error
	console_log("RMSE")
	RMSE	= sqrt( ( (diff - org)**2 ).mean() )					#Calculating RMSE
	return RMSE


################################################################################
#
# Compare altered image agains the original image
#
#	@param title: Title of altered image to compare	
################################################################################
def compare( title ):
	global genIMG
	metrics={}																						#Locale dict for metrics
	dImgTitle = title																			#Grab title
	dImgArr		= genIMG[title][0][:]												#Grab reversion array
	dImgImg		= genIMG[title][1][:]												#Grab image

	histOrg = calcHistogram( imgAction(oIMG, [0]) )				#Calc histogram org img
	histFil	= calcHistogram( imgAction(dImgImg, dImgArr))	#Calc histogram mod img
	histDif	= histFil[:]																	#Create list for diff
	for i in range(0,len(histOrg)):
		histDif[i] = abs(histOrg[i] - histFil[i])						#Populate diff list

	metrics.update({"RMSE": 															#Calculate and add
									rmse(imgAction(dImgImg, dImgArr),			#RMSE to metrics list
									oIMG)})																#

	#Plot histograms
	setSPlotSize([2,2]);																	#2x2 plot
	sPlot(spPos, [0], 	oIMG, "Original image BW")				
	sPlot(spPos, dImgArr, dImgImg, dImgTitle)	
	sPlot(spPos, [20], [[0, amax(histOrg)+500 ], histOrg, histFil], 
				"Histogram - both images")
	sPlot(spPos, [20], [[0, amax(histDif)+100 ], histDif],
				"Histogram - diff")
	plt.show()

	histComp(histOrg,histFil, metrics)										#Generate metrics

	metrics.update({"filHist":histFil})										#Add histogram 2 metrics
	metrics.update({"difHist":histDif})										#Add histogram 2 metrics
	genIMG[title].append(metrics)													#Add metrics to image


################################################################################
#
# Function for reversing a binary mask
#
#	@param	fltr:		A numpy array of MxN with a binary value of 0 or 1
#	@return rFltr:	Returns the reverse mask of filter
################################################################################
def reverseMask( fltr ):
	rFltr = fltr[:]
	for a in range(0,len(fltr)):
		for b in range(0, len(fltr[a])):
			rFltr[a][b] = 1 if fltr[a][b]==0 else 0
	return rFltr


################################################################################
#
# Normalise the image after filter has been applied.
#
# @param img: 	Image reverted from frequency to spatial doamn
# @return img:	Normalised image in the spatial domain
################################################################################
def normalizeGrayImg( img ):
	mi	= amin( img )																	#Grab lowest value
	ma	= amax( img )																	#Grab largest value
	sc	= ma / float(255)															#Calculate scale
	for a in range(0, len(img)):											
		for b in range(0, len(img[a])):
			img[a][b] = ((abs(img[a][b])-mi) / sc) - mi		#Normalise image
	return img																				#Return new image


################################################################################
#
# Function for choosing which filter to apply to an image
#
#	@return filters[choice]:	Returns the title of the chosen filter to apply
################################################################################
def chooseFilter( ):
	print "Available filters: "
	i = 0
	for f in filters:
		print "%2d: %s" % (i, f)
		i += 1
	choice = int( raw_input("\nChoose filter: ").rstrip() )
	return filters[choice]


################################################################################
#
# Applies a filter which is chosen to the original image
#
################################################################################
def applyFilter( D0=100 ):
	global genIMG
	fltr = chooseFilter()								#Choose filter to apply
	print "You chose the %s, please input data or use its default" % fltr
	
	iFFTS = genIMG["Shifted FFT"][1]		#Grab shifted FFT2 of orignal image
	M,N 	= iFFTS.shape									#Get the dimensions
	t 		= "ideal"											#Default filter
	n 		= 1.0													#btw-multiplier
	k			= 1.0
	Hf		= None

	happy = False
	print "Cut-off frequency as single integer or float greater than 0."
	D0	= float(raw_input("D0:\t"))		#grab Cut-off value
	while happy is False:
		nIMG = None																						#Zero out variable

		#Set filter parameters, and select butterworth multiplier
		if fltr == "LP: Buttwerworth":
			t = "btw"
			print "Butterwort multiplier value. Float value <= 1.0"
			n	= float(raw_input("n:\t"))
		elif fltr == "HP: Buttwerworth":
			print "Butterwort multiplier value. Float value equal to or greater than 1.0"
			n	= float(raw_input("n:\t"))
		elif fltr == "LP: Gaussian" or fltr == "HP: Gaussian":
			t = "gaussian"

		if fltr[0:2] == "LP":
			Hf 	= imgAction( genLPFilter(t,M,N,D0,n), [2] )			#LP filter
			fp	= (Hf*iFFTS)																		#Create filter (FFT)
			nIMG = imgAction( fp, [2,1] )												#revert to spatial

		elif fltr[0:2] == "HP":
			print "Multiplier value for the mask application. (k=%d)" %k
			k			= float(raw_input("k:\t"))										#grab k value
			Hf 		= imgAction( genLPFilter(t,M,N,D0,n), [2] )		#LP filter
			mask	= (iFFTS - (iFFTS*Hf))												#Generate mask
			fp		= (iFFTS + (mask * k))												#Image w/o LP
			nIMG	= oIMG + (k*imgAction(fp, [2,1]))							#Apply mask to image

	
		elif fltr == "Noise: Line rotate":
			degree = 0																					#Init rotation angle
			H = (abs(iFFTS[:]) * 0)															#Create temp mesh
			Hf= H[:]																						#Create result mesh
			Y,X = H.shape																				#Grab size
			for y in range(0, (Y/2)-int(D0)):										#Create binary mask
				for x in range((X/2)-1, (X/2)+1):									#of line filter
					H[y][x] = 1																			#Mark cell as 1

			print "Treshold value for mask application. (k=%.3f)" % k
			k			= float(raw_input("k:\t"))										#grab k value

			pxCount 	= (3*((Y/2)-D0))													#Mask area
			pxTot			= Y*X																			#Total number of px
			magTotal	= sum( abs(iFFTS) )												#Total magnitude
			pxPer			= pxCount / float(pxTot)									#Percentage filter is
			magLocal	= (magTotal * pxPer)											#Mag in area of filter

			while degree < 360:
				h = spMisc.imrotate(H, degree )										#Rotate line mask
				ni = (iFFTS * h)																	#Convolute ifft & mask
				magFilter	= sum( abs(ni) )												#Local magnitude
			
				if (magFilter / (magLocal*100)) >= k:
					Hf = (Hf + h)																		#Add filter to main
				degree = degree + 0.5															#increase rotation
			
			Hf = reverseMask( Hf )															#Reverse the mask
			fLP		= (Hf*iFFTS)																	#Convolve FFT & filter
			nIMG	= imgAction( (iFFTS + (1*fLP)) , [2,1,0] )		#Create new image

		elif fltr == "Noise: Horizontal-Vertical":
			Hf 		= (abs(iFFTS[:]) * 0)													#Create temp mesh
			Y,X 	= Hf.shape																		#Grab size
			Hx 		= [0,]*X																			#Row filter
			Hy 		= [0,]*Y																			#Column filter
			Hcx 	= Hf[:]																				#tmp filters
			Hcy		= Hf[:]																				#tmp fitlers
			pxTot	= X*Y																					#total pixel count

			for y in range(0, (Y/2)-int(D0)):										#Create binary mask
				Hy[y] 		= 1																			#Mark cell as 1
				Hy[Y-y-1] = 1																			#Mark cell as 1
			for x in range(0, (X/2)-int(D0)):										#Create binary mask
				Hx[x] 			= 1																		#Mark cell as 1
				Hx[X-x-1] 	= 1																		#Mark cell as 1
			
			magTotal	= sum( abs(iFFTS) )												#Total magnitude
			pxYPer		= Y / float(pxTot)												#Percentage filter is
			pxXPer		= X / float(pxTot)												#Percentage filter is
			magYLocal	= (magTotal * pxYPer)											#Mag in area of filter
			magXLocal	= (magTotal * pxXPer)											#Mag in area of filter

			print "Treshold value for mask application. (k=%.3f)" % k
			k			= float(raw_input("k:\t"))										#grab k value

			#Vertical filter line
			for x in range(0,X):																#Generate vertical 
				for y in range(0,Y):															#filter for all cols
					Hcy[y][x] = Hy[y]																#
				ni = (iFFTS * Hcy)																#Convolute ifft & mask
				magFilter	= sum( abs(ni) )												#Local magnitude
				comp = magFilter/(magYLocal*100)									#
				if comp >= k:
					Hf = (Hf + Hcy)																	#Add filter to main

			#Horisontal filter line
			for y in range(0,Y):																#Generate horisontal
				for x in range(0,X):															#filter for all rows
					Hcx[y][x] = Hx[x]																#Set tmp filter pixel
				ni = (iFFTS * Hcx)																#Convolute ifft & mask
				magFilter	= sum( abs(ni) )												#Local magnitude
				comp = magFilter/(magXLocal*100)									#grab comparison value
				if comp >= k:																			#Check requirements
					Hf = (Hf + Hcx)																	#Add filter to main

			fLP		= (Hf*iFFTS)																	#Convolve FFT & filter
			nIMG	= imgAction( (iFFTS + (1*fLP)) , [2,1,0] )		#Create new image

		elif "Noise: Local spots":
			iFFTS2 		= abs(iFFTS)															#Absolute of FFT2
			Hf 				= (iFFTS2 * 0)														#Create temp mesh
			Y,X 			= Hf.shape																#Grab size
			
			print "Treshold value for mask application (avg*k). (k=%.3f)" % k
			k			= float(raw_input("k:\t"))										#grab k value

			for y in range(2,Y-3):															#Grab mean of 5 cells
				for x in range(2,X-3):
					comp = (iFFTS2[y][x-1] + iFFTS2[y][x+1] + iFFTS2[y][x] + 
									iFFTS2[y-1][x] + iFFTS2[y+1][x] ) / float(5)
					Hf[y][x] = comp																	#Populate w/comp value
			avg = sum(Hf)/float(X*Y)														#Grab average
			
			for y in range(0,Y):																#Check if cell value
				for x in range(0,X):															#is k times greather
					if Hf[y][x] > avg*k:														#than the average
						Hf[y][x] = 0																	#	then block out cell
					else:
						Hf[y][x] = 1																	# else allow it
				Hf[y][0] = 1																			#Set left side static-
				Hf[y][1] = 1																			#ally
			Hf = Hf + imgAction(genLPFilter("ideal",M,N,D0,n), [2])	#LP filter
			
			nIMG = imgAction( (iFFTS*Hf),	[2,1,0])							#image + mask

		else:
			print "Unknown Filter, quitting"										#print error and
			return																							#Escape function
		
		nIMG = normalizeGrayImg(nIMG)													#image + mask
		setSPlotSize([1,3])																		#Set image layout
		sPlot(spPos, [],			(oIMG),			"Original image")		#image + mask
		sPlot(spPos, [0],			nIMG,				"New Image")				#image + mask
		if fltr[0:2] in ["LP","HP"]:
			sPlot(spPos, [3,0],		(fp),	"Applied filter")				#image + mask
		else:
			sPlot(spPos, [3,0],		(Hf*iFFTS),	"Applied filter")	#image + mask
		plt.show()																						#Show image

		print "Happy with current filter, (D0=%d, n=%.2f, k=%d)?" % (D0, n, k)
		print "Available actions:  -X, +X. Y, Q"
		ans = raw_input("Action: ")														#Decide what to do
		if ans == "Y":																				#next
			happy = True
			genIMG.update({fltr+" [D0=%d n=%.2f k=%d]" %(D0,n,k):
										[[0], nIMG ]})
		elif ans == "Q":
			print "Exiting"
			happy = True
		elif ans == "":
			pass
		elif ans[0] == "-":
			print "Subtracting %s from cutoff freq (D0=%d)" % (ans[1:], D0)
			D0 = D0 - int(ans[1:])
		elif ans[0] == "+":
			print "Adding %s to cutoff freq (D0=%d)" % (ans[1:], D0)
			D0 = D0 + int(ans[1:])


################################################################################
#
# Save image and data
#
# @param title: Title of image with metrics
################################################################################
def saveImage( title ):
	print "Saving image to file. (./outdir/"+title+".jpg)"
	spMisc.imsave("./outdir/"+title+".jpg",								#Save image to file
								imgAction(genIMG[title][1], genIMG[title][0]))

	if len(genIMG[title]) >=3:														#if img contains metrics
		console_log("Saving image metrics to file.")				#
		mets	= genIMG[title][2]														#grab metrics
		out = open("./outdir/"+title+".met", "w")						#open file
		for t in sorted(mets.keys()):												#
			out.write( "%25s: %s" % (t, mets[t]) )						#Write metrics to file
		out.close()																					#Close file
	else:																									#
		console_log("No metrics stored for this image.")		#Log error message


################################################################################
#
# Print metrics
#
# @param title: Title of image with metrics
################################################################################
def dispMetrics( title ):
	if len(genIMG[title]) < 3:														#If metrics don't exists
		print "No metrics for this image exists. Please run a comparison."
		return
	console_log("Displaying image metrics.")
	mets = genIMG[title][2]																#Grab title
	for t in sorted(mets.keys()):													#
		if t not in ["difHist","filHist"]:									#avoid histograms
			print "%35s: %s" % (t, mets[t])										#Print metrics








################################################################################
################################################################################
##
##	Main running script
##
################################################################################
################################################################################


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
fIMG	= sys.argv[1]																	#Grab filename
oIMG	= imread( fIMG, 0 )														#Load image as grayscale
genIMG.update({"Original image BW":[[0],oIMG]})			#save image in image info
splitThatImage()																		#Split image into

a = genIMG["Shifted FFT"][1]

QUIT = False
while QUIT is False:
	print "%s\n##\tMenu options\n%s\n" % ("#"*80, "#"*80)
	print "\t1 - Display a set of images"
	print "\t2 - Perform filter operations on image"
	print "\t3 - Compare an altered image against the original and show the data"
	print "\t4 - Display metrics of images"
	print "\tS - Save new image to file"
	print "\tQ - Save data and quit the program"
	print ""
	ans = raw_input("Choice:..... ")
	if ans == "1":																		#Show a set of images
		showImages()

	elif ans == "2":																	#Apply filter to image
		applyFilter( 100 )

	elif ans in ["3", "4", "S"]:											#Compare, display or save
		titles = chooseImages()
		print "Following images was selected: " + ", ".join(titles)
		for i in titles:
			if ans == "3":																#Compare and gen metrics
				compare( i )
			elif ans == "4":															#Display metrics
				dispMetrics( i )
			elif ans =="S":																#Save images to file
				saveImage( i )

	elif ans == "Q":																	#QUIT
		for i in genIMG.keys():													#Save all images and metrics
			saveImage( i )
		print "Exiting the program without further actions"
		QUIT = True
		
	else:																							#Do another run
		pass



