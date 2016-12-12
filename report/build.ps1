######################################################
# Parameters for build.ps1 is "-length" or "-l"
######################################################
# quick			-	Builds the report once
#	double		-	Builds the report twice
# regular		- Builds the report three times with bibliography
# full			- Like regular, but with glossary, definitions in addition
# all				- Builds everything, including listings
######################################################
Param(
	[parameter(Mandatory=$FALSE)]
    [alias("l")]
    $length="double"
)

#############################
# Variable declarations
#############################
$AUX_FILES_LIST = @("*.aux", "*.bbl", "*.blg", "*.bst", "*.glg", "*.glo", "*.gls", "*.ilg", "*.ist", "*.lof", "*.lol", "*.lot", "*.nlo", "*.nls", "*.out", "*.tdo", "*.toc", "*.xdy", "main.pdf")

#############################
# Function declarations
#############################
function clean {
	Param($meth="reg")
	
	ForEach( $EL in $AUX_FILES_LIST) {
		if( $TRUE -eq $(Test-Path $EL) ){
			remove-item $EL
		}
	}

	if( $meth -eq "all" ){
		if( $TRUE -eq $(Test-Path "pictures/*-converted-to.pdf") ){
			remove-item remove-item pictures/*-converted-to.pdf
		}
	}
}

function build {
	pdflatex main.tex
}

function bib {
	bibtex main.aux
}

function glosarries {
	makeglossaries.exe main.glo
	makeindex main.nlo -s nomencl.ist -o main.nls
}


#############################
# MAIN RUNNER
#############################
if( $length -eq "quick" -OR $length -eq "double" ) {
	build
	if( $length -eq "double" ) {
		build
	}
}
else if( $length -eq "regular" ) {
	clean
	build 
	build
	bib
	build
}
else if( $length -eq "full" -OR $length -eq "all"){
	clean "all"
	build
	build
	bib
	glossaries
	build
}
else if( $length -eq "clean" ){
	clean
}
else if( $length -eq "wipe" ){
	clean "all"
}
else{
	write-host "Parameter: -length (-l) followed by one of the following arguments"
	write-host "quick			-	Builds the report once"
	write-host "double		-	Builds the report twice"
	write-host "regular		- Builds the report three times with bibliography"
	write-host "full			- Like regular, but with glossary, definitions in addition"
	write-host "all				- Builds everything, including listings"
	write-host "===================================================================="
	write-host "clean			- Cleans most of the additionally created files"
	write-host "wipe			- Cleans also the generated PDF files in ./pictures/"
}
