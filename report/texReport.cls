%% 'gucreport.cls' - a LaTeX class for reports at GUC, based upon gucthesis.cls
%%
%% Copyright (C) 2014 Magnus Øverbø
%% Based on the copyrighted template - 'gucthesis.cls' - 
%%	'gucthesis-cls' - Copyright (C) 2005 Ivar Farup and Kjetil Orbekk
%%
%% 2014/02/06
%%	* Removed redundant code from creating front pages
%%	* Specific types of study programs are more easily created
%%	* Added a separate counter for wordcount
%%	* Added a second title page for bachelor students
%%	* Fixed norwegian/english language bug
%%	*	Fixed cleardoublepage on summaries on twoside
%%
%% 2014/02/23
%%	* Created report template with frontpage 
%%	* 

%%	Fix norsk option, since it returns errors on UTF8x
%%	


%%
%%	IF variables
%% % % % % % % % % %
\newif\if@norsk				\@norskfalse
\newif\if@english			\@englishtrue
\newif\if@showIndex		\@showIndexfalse
\newif\if@showDict		\@showDictfalse
\newif\if@showTodo		\@showTodofalse
\newif\if@showBib			\@showBibfalse

%%
%% IDENTIFICATION
%% % % % % % % % % %
\xdef\srepd{2014/02/28}
\xdef\srepv{1.00.00}
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{texReport}
  [\srepd\space
   v\srepv\space
   Copyright (C) 2014 Magnus Øverbø.
	 Based on the gucthesis class by Ivar Farup and Simon McCallum]

%%
%% CLASS FILE COMMANDS
%% % % % % % % % % %
\LoadClass[a4paper,10pt]{report}

%%
%% PACKAGE LOADING
%% % % % % % % % % %
\RequirePackage{geometry}
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8x]{inputenc}
\RequirePackage{euler}
\RequirePackage{lastpage}
\RequirePackage{babel}
\RequirePackage[table]{xcolor}
\RequirePackage[pdftex]{graphicx, hyperref}
\RequirePackage{pdfpages}
\RequirePackage{color, multicol, longtable, amsmath, amssymb}
\RequirePackage[all]{hypcap}
\RequirePackage[acronym, nomain]{glossaries}
\RequirePackage{glossary-mcols}
\RequirePackage{listings, bbding, caption, todonotes}

%\setglossarystyle{mcolindex}
\setglossarystyle{mcolindex}
\setacronymstyle{long-short}


%%
%%	Counter variables
%% % % % % % % % % %
\newcounter{firstchapter}       % hack to find out where to start
	\c@firstchapter=1             % arabic page numbering, see below
\newcounter{tmpfig}             % hack to have continuous numbering of
\newcounter{tmptab}             % figures and tables, see below
\newcounter{numapp}


%%
%%	Class Options
%% % % % % % % % % %
\DeclareOption{norsk}{
	\@norsktrue \@englishfalse
}
\DeclareOption{english}{
	\@englishtrue	\@norskfalse
}
\DeclareOption{oneside}{
	\@twosidefalse \@mparswitchfalse%
	\geometry{a4paper, left=3.75cm, right=3.75cm, top=3cm, 
						bottom=4cm, head=1.2cm, foot=2cm}
}
\DeclareOption{twoside}{
	\@twosidetrue  \@mparswitchtrue%
	\geometry{a4paper, left=4.5cm, right=3cm, top=3cm, 
						bottom=4cm, head=1.2cm, foot=2cm}
}
\DeclareOption{listings}{
	\@showIndextrue
}
\DeclareOption{glossary}{
	\@showDicttrue
}
\DeclareOption{todo}{
	\@showTodotrue
}
\DeclareOption{bib}{
	\@showBibtrue
}

%%
%%	Definistions / Variables
%%		Elements used by the template which can be changed by the user
%% % % % % % % % % %
\def\mainTitle{ mainTitle }
\def\mainTitle#1{ \def\mainTitle{#1} }
\def\subTitle{ subTitle }
\def\subTitle#1{ \def\subTitle{#1} }

\def\courseId{ Course ID }
\def\courseId#1{\def\courseId{#1}}
\def\courseTitle{ Course Title }
\def\courseTitle#1{\def\courseTitle{#1}}

\def\orgNameNo{ N/A }
\def\orgNameNo#1{ \def\orgNameNo{#1} }
\def\orgNameEn{ N/A }
\def\orgNameEn#1{ \def\orgNameEn{#1} }

\def\deptNameNo{ N/A }
\def\deptNameNo#1{ \def\deptNameNo{#1} }
\def\deptNameEn{ N/A }
\def\deptNameEn#1{ \def\deptNameEn{#1} }

\def\supervisor{ Supervisor/Team Lead }
\def\supervisor#1{\def\supervisor{#1}}

\def\logo{ntnu_wide_en}
\def\logo#1{\def\logo{#1}}

\def\wordcount{ N/A }
\def\wordcount#1{\def\wordcount{#1}}

\def\pDate{ \today }
\def\pDate#1{ \def\pDate{#1} }

\def\keywords{ Keywords }
\def\keywords#1{ \def\keywords{#1} }

\def\appnumber{ Appnumber }
\def\appnumber#1{ \def\appnumber{#1} }

\def\pagecount{ LastPage  }
\def\pagecount#1{ \def\pagecount{#1} }

\def\pVersion{ Version }
\def\pVersion#1{ \def\pVersion{#1} }

\def\projauthorid{ Author id}
\def\projauthorid#1{\def\projauthorid{#1}}
\def\projauthor{ Author }
\def\projauthor#1{\def\projauthor{#1}}

\def\projauthorAid{Author ID}
\def\projauthorAid#1{\def\projauthorAid{#1}}
\def\projauthorA{Author Name }
\def\projauthorA#1{\def\projauthorA{#1}}

\def\projauthorBid{Author ID}
\def\projauthorBid#1{\def\projauthorBid{#1}}
\def\projauthorB{Author Name }
\def\projauthorB#1{\def\projauthorB{#1}}

\def\projauthorCid{Author ID}
\def\projauthorCid#1{\def\projauthorCid{#1}}
\def\projauthorC{Author Name }
\def\projauthorC#1{\def\projauthorC{#1}}

\def\projauthorDid{Author ID}
\def\projauthorDid#1{\def\projauthorDid{#1}}
\def\projauthorD{Author Name }
\def\projauthorD#1{\def\projauthorD{#1}}


%%
%%	Functions
%% % % % % % % % % % % % %
\newcommand{\numberofapp}{%
	\immediate\write\@auxout%
	{\string\setcounter{numapp}{\the\c@chapter}}%
}

%Prints information about course
\newcommand{\courseInfo}{	
	{\large \courseId --- \courseTitle }
	%\textbf{\large \courseId --- \courseTitle }
}

%returns name/path of the org logo
\newcommand{\getLogo}{pics/\logo}

%prints info about the org and dept.
\newcommand{\deptName}{%
	\if@norsk 	\deptNameNo \\ \orgNameNo
	\else				\deptNameEn \\ \orgNameEn
	\fi
}

%Generates appropriate spacing based on open right rule
\newcommand{\pagespace}{
	\if@openright		\cleardoublepage
	\else						\clearpage
	\fi
}

%%
%%	CUSTOM CONFIG AND COMMANDS
%% % % % % % % % % % % % %
\renewcommand*{\glspostdescription}{}
\renewcommand{\lstlistingname}{Code}
\renewcommand\lstlistlistingname{\if@norsk Kodesnutter og skript
	\else	Code snippets and scripts\fi
}
\newcommand{\comment}[1]{\textcolor{blue}{\emph{#1}}}	%	Comment
\newcommand{\com}[1]{{\color{red}#1}}									% Supervisory comment
\renewcommand{\todo}[1]{{\color{green}#1}}						% Items to do
\newcommand{\n}[1]{{\color{blue}#1}} 									% Other comment
\newcommand{\dn}[1]{} 																% Finished comment


%%
%%	Rename commands and create new glossary "default, in addition to "acronym"	
%% % % % % % % % % % % % %
\newcommand{\nge}[4]{ 	\newglossaryentry{#1}{name={#2}, plural={#3}, description={#4}} }
\newcommand{\lnge}[4]{ 	\longnewglossaryentry{#1}{name={#2}, plural={#3}}{#4} 	}
\newcommand{\nacr}[3]{	\newacronym{#1}{#2}{#3}				}

\newglossary[nlg]{default}{not}{ntn}{Glossary}

%%
%%	Load glossary files from disk
%% % % % % % % % % % % % % % % % % % % % % % %
\loadglsentries[default]{glossary_default}
\loadglsentries[acronym]{glossary_acronym}

\setlength{\columnsep}{3em}											% Set column separation distance

\AtBeginDocument{
	\baselineskip=14pt%
  \parskip=0pt%
  \parindent=14pt%
  \frenchspacing%
  \setcounter{secnumdepth}{2}%
  \setcounter{tocdepth}{1}%
	\if@showDict
		\makeglossaries
	\fi
}
\pagenumbering{Roman}           % until first chapter, see below
\captionsetup{justification=centering}

%% Headings
\def\ps@document{%
	\def\@oddfoot{\reset@font \hfill \thepage \hfill }
  \let\@evenfoot=\@oddfoot
  \def\@oddhead{
		\underline{
			\hbox to\hsize{
				\fontsize{8}{10}	\selectfont \hfill \projtitle \hfill
			}%hsize
		}%underline
	}%oddhead
  \let\@evenhead=\@oddhead
}

\def\ps@preamble{%
  \def\@oddhead{}%oddhead
  \let\@evenhead=\@oddhead
}

% Empty page does not have header or footer
\def\cleardoublepage{
	\clearpage
	\if@twoside
		\ifodd	\c@page
		\else		\hbox{}\thispagestyle{empty}\newpage
			\if@twocolumn	\hbox{}\newpage
			\fi
		\fi
	\fi
}

% Title page
\def\projtitlepage{
	\title{	\projtitle	}
  \date{	\pDate	}
  \author{\projauthor}
  \maketitle
}

% Sectioning commands
\def\section{\@startsection
	{section}	{1}	{0mm}	{-10pt}	{4pt}
  {\normalfont \fontsize {12} {14} \selectfont \bfseries}
}
\def\subsection{\@startsection
  {subsection}	{2}	{0mm}	{-6pt}	{2pt}%
  {\normalfont\fontsize{10.5}{14}\selectfont\bfseries}%
}
\def\subsubsection{\@startsection
  {subsubsection}	{3}	{0mm}	{-6pt}	{2pt}%
  {\normalfont\normalsize\selectfont\bfseries}%
}
\def\paragraph{\@startsection
  {paragraph}	{4}	{0mm}	{-6pt}	{2pt}%
  {\normalfont\normalsize\selectfont\itshape}%
}

\renewcommand\chapter{
	\pagespace
  \global\@topnum\z@
  \@afterindentfalse
  \secdef\@chapter\@schapter
}
	\def\@chapter[#1]#2{\ifnum \c@secnumdepth >\m@ne
  \c@tmpfig=\c@figure           % hack for figure and table numbering
  \c@tmptab=\c@table
  \refstepcounter{chapter}%
  \c@figure=\c@tmpfig
  \c@table=\c@tmptab
 	\ifnum\c@firstchapter = 1     % hack for page numbering
  \pagenumbering{arabic}
  \c@page=1 \c@firstchapter=0
  \fi
  \typeout{\@chapapp\space\thechapter.}%
  \phantomsection
  \addcontentsline{toc}{chapter}%
  {\protect\numberline{\thechapter}#1}%
  \else
  \phantomsection
  \addcontentsline{toc}{chapter}{#1}%
  \fi
  \chaptermark{#1}%
  \if@twocolumn
  \@topnewpage[\@makechapterhead{{#2}}]%
  \else \@makechapterhead{{#2}}%
  \@afterheading
  \fi}
\def\@makechapterhead#1{%
  \vspace*{24\p@}%
  {\parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
    \begin{center}
    \normalfont\fontsize{14}{14}\selectfont\bfseries\thechapter
    \fi
    \normalfont\fontsize{14}{14}\selectfont\bfseries\quad #1
    \end{center}\par\nobreak
    \vskip 12\p@
  }
}
\def\@schapter#1{\if@twocolumn
  \@topnewpage[\@makeschapterhead{{#1}}]%
  \else
  \@makeschapterhead{{#1}}%
  \@afterheading
  \fi
  \phantomsection
  \addcontentsline{toc}{chapter}{#1}% Contentsline also for \chapter*
}
\def\@makeschapterhead#1{%
  \vspace*{24\p@}%
  {\parindent \z@ \raggedright
    \normalfont
    \interlinepenalty\@M
    \begin{center}\fontsize{14}{14} \bfseries  #1\end{center}\par\nobreak
    \vskip 12\p@
  }}

% Table of contents
\def\l@chapter#1#2{\@dottedtocline{1}{0em}{1.5em}{\bf #1}{\bf #2}}

% Table and figure captions
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{\fontsize{9}{13}\selectfont #1: #2}%
  \ifdim \wd\@tempboxa >\hsize
    \fontsize{9}{13}\selectfont #1: #2\par
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip}

% Table and figure numbering without chapter number
\def\thefigure{\@arabic\c@figure}
\def\thetable{\@arabic\c@table}

% Quotes
\renewenvironment{quote}
{\list{}{\rightmargin\leftmargin\fontsize{9}{12}\selectfont}%
\item\relax}
{\endlist}

% Lists (itemize and enumerate)
\renewenvironment{itemize}{\begin{list}{\ensuremath\bullet}%
    {\labelwidth=.5em%
      \labelsep=1em%
      \leftmargin=\labelwidth%
      \advance\leftmargin\labelsep%
      \rightmargin=0pt%
      \topsep=5pt%
      \parsep=0pt%
      \partopsep=\baselineskip%
      \itemsep=\topsep%
      }}{\end{list}}

\newcounter{nenum}
\renewenvironment{enumerate}{\begin{list}{\llap{\arabic{nenum}.}}%
     {\usecounter{nenum}%
      \labelwidth=.7em%
      \labelsep=.8em%
      \leftmargin=\labelwidth%
      \advance\leftmargin\labelsep%
      \rightmargin=0pt%
      \topsep=5pt%
      \parsep=0pt%
      \partopsep=0pt%
      \itemsep=\topsep%
      }%
   }{\end{list}}


%%
%%	MACRO
%% % % % % % % % % % % % %
\newcommand{\gmt@frontpagesyear}{%
	\textbf{\color{red}N/A - \textbackslash useyear}
}
\newcommand{\useyear}[1]{%
  \renewcommand{\gmt@frontpagesyear}{#1}
}

%%
%%
%% % % % % % % % % % % % % % % % % % % % % % %
\newcommand\showList{
	\pagespace
		\let\pagespace\relax
		
		\begingroup
			\if@showIndex
				\listoffigures
				\lstlistoflistings
				\listoftables
			\fi
			\if@showDict 	
				\printglossary[type=default, title={Glossary}]
				\printglossary[type=acronym, title={Abbreviations}]
			\fi
			\if@showTodo	\listoftodos			\fi
		\endgroup
	\pagespace
}

%%
%%	Command to create and show bibliography and add appendices
%% % % % % % % % % % % % % % % % % % % % % % %
\newcommand\makebib{
	\if@showBib
		% plain, unsrt, alpha, abbrv
		\bibliography{citation}				%citation.bib
		%\bibliographystyle{abbrv}
		\bibliographystyle{acm}
		%\bibliographystyle{alpha}
		%\bibliographystyle{apalike}
		%\bibliographystyle{ieetr}
		%\bibliographystyle{plain}
		%\bibliographystyle{siam}
		%\bibliographystyle{unsrt}
		%\bibliographystyle{}
		%\bibliographystyle{}
	\fi
	\pagespace
	\appendix
	\input{appendices}
}

\ProcessOptions\relax

%% FRONTPAGE GENERATION
\newcommand\makefrontpages{
	\begin{titlepage}
		\null
		\vskip 4mm
    \begin{center}
			\underline{\Huge \mainTitle }\\[4mm]
			{\huge \color[gray]{0.2} \subTitle }\\[13mm]

			\hskip 20mm
			\begin{tabular}[2]{ p{ 0.25\textwidth} p{0.7\textwidth} }
				{\Large \projauthorid}	&	{\Large \projauthor} 	\\[2mm]
				{\Large	\projauthorAid}	&	{\Large \projauthorA} \\[2mm]
				{\Large \projauthorBid} &	{\Large \projauthorB} \\[2mm]
				{\Large \projauthorCid} &	{\Large \projauthorC} \\[2mm]
				{\Large \projauthorDid} &	{\Large \projauthorD} \\[2mm]
			\end{tabular}\\[5mm]
				
			\hskip 20mm
			\begin{tabular}[2]{ p{0.25\textwidth}  p{0.7\textwidth} }
				{\Large	\if@norsk	Dato:	\else	Date:\fi	}	&	
				{\Large \pDate }\\[1mm]
				
				{\Large	\if@norsk	Versjon:	\else	Version:\fi	}	&	
				{\Large	\pVersion }\\[1mm]
				
				{\Large	\if@norsk	Ansvarlig:	\else	Supervisor:\fi }	&	
				{\Large	\supervisor }\\[1mm]
				
				{\Large	\if@norsk	Sideantall: \else Page count: \fi	} &	
				{\Large	\pagecount }\\[1mm]
				
				{\Large	\if@norsk	Antall ord: \else Word count: \fi	} &	
				{\Large	\wordcount }\\[1mm]
				
				{\Large	\if@norsk	Vedlegg: \else Appendices: \fi } &
				{\Large \appnumber }\\[1mm]
				
				{\Large	\if@norsk	N\o{}kkelord: \else Keywords: \fi	} &	
				{\Large \keywords }
			\end{tabular}\\[5mm]
    \end{center}
		
		\vfill
		\begin{center}
			\includegraphics[width={10cm}]{\getLogo}\\[2mm]
			\courseInfo \\{	\large	\deptName }\\
    \end{center}
	\end{titlepage}
	\pagespace
}


%	Package setup
\definecolor{commentfg}	{rgb}{0.60,	0.00,	0.00}
\definecolor{stringfg}	{rgb}{0.25,	0.35,	0.75}
\definecolor{codebg}		{rgb}{1.00,	1.00,	0.90}

\hypersetup{
	colorlinks=false,
	pdfborder={0 0 0}
}

\lstset{extendedchars=\true}
\lstset{inputencoding=utf8x}

\lstset{
	language					= Python, %%Default language
	numbers		  			= left,
	breaklines				= true,
	breakindent				= 10pt,
	breakautoindent 	= true,
%	extendedchars			= true,
	stepnumber				= 1,
	numbersep	  			= 5pt,
	numberstyle				= \footnotesize \color{black},
	basicstyle				= \footnotesize \ttfamily,
	keywordstyle			= \footnotesize \color{black},
	commentstyle			= \footnotesize \color{commentfg},
	stringstyle				= \footnotesize\color{stringfg},
	showstringspaces	= false,
	frame		    			= none,
	tabsize		  			= 2,
	backgroundcolor		=	\color{codebg}
}  
%%Languages
%	Lang    	Name			Dialects
%------- 	--------	------------------->
% Python		Python		
%	SQL				SQL
%	PHP				PHP
% C					C					ANSI, Objective, Sharp
% C++				C++				ANSI, GNU, ISO, VISUAL
%	Basic			Basic			Visual
%	HTML			HTML
% Lua				Lua				5.0,	5.1,	5.2
%	Matlab		Matlab
%	Bash			bash
%	sh				sh
%	XML				XML
%	TeX				TeX				common,	LaTeX, plain, primitive
%	VBScript	VBScript
%	Java			Java			empty, AspectJ
% Assmebler	Assembler	x86masm

