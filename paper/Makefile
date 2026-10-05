# Build the paper.
#
#   make            -> full build (pdflatex, bibtex, pdflatex x2)
#   make quick      -> single pass, for checking a figure or table fast
#   make test       -> build WITHOUT Slicing.bib / IEEEabrv.bib / the bio
#                      photo, i.e. what this repo can build on its own.
#                      Citations render as [?]. Use this to check figures.
#   make clean      -> remove build artefacts
#
# Requires a TeX distribution providing pgfplots, tikz and IEEEtran.
# On macOS:  brew install texlive
#
# To build the real paper you need, in this directory:
#   Slicing.bib, IEEEabrv.bib      (your bibliography)
#   Figures/Amaral_3x4.jpg         (biography photo)
# and the entries from Slicing_additions.bib appended to Slicing.bib.
#
# Figures are external PDFs in figures/ and are already built.
# Regenerate the data plots:      python3 make_figures.py
# Regenerate the schematic:       cd figures && pdflatex fig_architecture.tex

DOC = NetworkSlicing_v3
LATEX = pdflatex -interaction=nonstopmode -file-line-error

.PHONY: all quick test clean

all:
	$(LATEX) $(DOC).tex
	-bibtex $(DOC)
	$(LATEX) $(DOC).tex
	$(LATEX) $(DOC).tex
	@echo "--- warnings worth checking ---"
	@grep -E "Reference.*undefined|Citation.*undefined" $(DOC).log | sort -u | head -20 || true
	@echo "--- overfull boxes over 20pt ---"
	@grep "Overfull \\\\hbox" $(DOC).log | awk -F'(' '{print $$2}' | awk -F'pt' '$$1>20' | wc -l

quick:
	$(LATEX) $(DOC).tex

# Self-contained build: stubs the assets that live outside this repo.
test:
	@mkdir -p .testbuild
	@sed 's|\\begin{IEEEbiography}\[{\\includegraphics.*}\]|\\begin{IEEEbiography}|' \
	    $(DOC).tex > .testbuild/$(DOC).tex
	@: > .testbuild/Slicing.bib; : > .testbuild/IEEEabrv.bib
	@mkdir -p .testbuild/figures && cp figures/*.pdf .testbuild/figures/
	@cd .testbuild && $(LATEX) $(DOC).tex >/dev/null 2>&1 && \
	  $(LATEX) $(DOC).tex >/dev/null 2>&1; \
	  if [ -f $(DOC).pdf ]; then \
	    cp $(DOC).pdf ../$(DOC)_testbuild.pdf; \
	    echo "OK -> $(DOC)_testbuild.pdf ($$(pdfinfo $(DOC).pdf | awk '/Pages/{print $$2}') pages)"; \
	  else echo "BUILD FAILED - see .testbuild/$(DOC).log"; fi
	@grep -E "^!" .testbuild/$(DOC).log | head || true

clean:
	rm -f $(DOC).aux $(DOC).log $(DOC).out $(DOC).bbl $(DOC).blg $(DOC).toc
	rm -rf .testbuild
