# Makefile for the "Better Code, Better Science" code repository.
#
# This repo holds the public code examples, notebooks, the tex2qmd pipeline,
# and the rendered web edition under docs/ (served by GitHub Pages).
#
# The book prose lives in a SEPARATE PRIVATE repo, cloned into ./latex locally.
# Targets that read ./latex (figures, pdf, site) require it to be present, plus
# pandoc, quarto, and pdftocairo on PATH for the web build.

.PHONY: help install test \
        figures figures-notebooks figures-gray figures-check \
        pdf verify-listings \
        site site-preview clean

help:
	@echo "Common targets:"
	@echo "  make install        Install dependencies and this package (uv)"
	@echo "  make test           Run the test suite with coverage"
	@echo "  make figures        Regenerate all book figures (notebooks + gray fill)"
	@echo "  make figures-check  Verify every referenced figure has a gray variant"
	@echo "  make pdf            Build the print PDF (delegates to latex/Makefile)"
	@echo "  make site           Build the Quarto web edition into docs/"
	@echo "  make site-preview   Live-preview the web edition"

# --- Setup --------------------------------------------------------------
install:
	uv pip install -r pyproject.toml
	uv pip install -e .

# --- Tests --------------------------------------------------------------
test:
	uv run pytest --cov=src/bettercode --cov-report=term-missing

# --- Figures ------------------------------------------------------------
# Notebooks write color figures to latex/figures/<chapter>/ (pdf + svg) and
# grayscale variants to latex/gray/figures/<chapter>/ (pdf) via save_fig /
# save_fig_variants. build_gray_figures then fills the remaining gray figures
# (hand-made diagrams, screenshots) that no notebook produces.
figures: figures-notebooks figures-gray

figures-notebooks:
	uv run python scripts/run_notebooks.py

figures-gray:
	uv run python scripts/build_gray_figures.py

figures-check:
	uv run python scripts/build_gray_figures.py --check
	uv run python scripts/figure_atlas.py --check

# --- Print PDF (book prose lives in ./latex) ----------------------------
pdf:
	cd latex && $(MAKE) all

verify-listings:
	cd latex && $(MAKE) verify-listings

# --- Quarto web edition (GitHub Pages) ----------------------------------
# The rendered site is committed under docs/ and served by Pages from main /docs.
site: verify-listings
	uv run python scripts/tex2qmd/convert.py --latex-dir latex --out web
	cd web && quarto render
	touch docs/.nojekyll   # stop GitHub Pages' Jekyll from mangling Quarto assets

site-preview:
	uv run python scripts/tex2qmd/convert.py --latex-dir latex --out web
	cd web && quarto preview

# Remove the intermediate Quarto project (docs/ is the committed output).
clean:
	- rm -rf web
