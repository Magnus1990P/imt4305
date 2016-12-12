CC1			= rm -f
CC2			= pdflatex
CF1			= -file-line-error 
LOG			= >> build.log
FILE		=	main
REM1		=	*.aux *.bbl *.blg *.bst *.glg *.glo *.gls *.glsdefs *.ilg *.ist 
REM2		= *.acn *.acr *.alg *.lof *.lol *.lot *.nlg *.nlo *.nls *.not *.ntn 
REM3		=	*.out *.tdo *.toc *.xdy 
REM4		= *.log ${FILE}.pdf

##		COMPILING FUNCTIONS
build: 
	${CC2} ${CF1} ${FILE}.tex #${LOG}
double:
	${CC2} ${CF1} ${FILE}.tex #${LOG}
	${CC2} ${CF1} ${FILE}.tex #${LOG}

##		Full and new compilation
small:	clean	double
normal:	clean build bib double
full:		clean build index double

##		Index/listing builds
index:	bib glossary
bib:
	bibtex main
glossary: 
	makeglossaries main 

##		CLEANING FUNCTIONS
##	Cleans almost all files
clean:	preclean	postclean
##	Cleans the remaining files
preclean:	
	${CC1} ${REM4} ${LOG}
##	Cleans log files and other unnessecary files
postclean:
	${CC1} ${REM1} ${REM2} ${REM3} ${LOG}
##	Removes all
wipe:		clean
	rm -f pics/*-converted-to.pdf
	rm -f build.log

