texReport.git
=============

Copyright 2014 by Magnus Øverbø,
Based on the copyrighted gucthesis.cls by Ivar Farup and Simon McCallum at NTNUiG.

This template is meant to ease the work around creating a fully formatted
report from the ground.

------------------------
Explanation:
------------------------
/chapters/ - is meant to hold all of the chapters and additional tex documents 
being created throughout of the report.

/pics/ - is meant to hold all pictures in the report, this results in a
shorter path needed to reference the image.

/chapters/appendices/ - Is meant to hold all appendices in the report.

/appendices.tex - Is used to reference all appendices in the the document, which
yields an easy overview of the appendices.

/citation.bib - Is the file which holds all references in bibtex format.

/glossary_acronym.tex - Contains just the acronyms and initialisms for the
report.

/glossary_default.tex - Is the main file for the glossary of the report except
for the content of Acronyms and Initialisms.

/makefile - This is the file which on linux eases the build process of the
report. Build the report using "make" in the terminal

/texReport.tex - This file holds all of the administrative information like
organization name, course name/id, number of appendices, authors, titles,
keywords, etc.

/texReport.cls - This file holds the configurations of the texReport class

------------------------
Requirements:
------------------------
+ texlive-full
+ pdflatex
+ makeglossaries
+ bibtex

