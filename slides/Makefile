TEX = pdflatex
FLAGS = -interaction=nonstopmode -halt-on-error
SRC = slides.tex
PDF = slides.pdf

.PHONY: all clean

all: $(PDF)

$(PDF): $(SRC)
	$(TEX) $(FLAGS) $(SRC)
	$(TEX) $(FLAGS) $(SRC)

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb $(PDF)
