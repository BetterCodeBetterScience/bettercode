MAIN = book
LATEXMK = latexmk
LATEXMK_FLAGS = -xelatex -f # -pdf -interaction=nonstopmode -halt-on-error

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex verify-listings
	$(LATEXMK) $(LATEXMK_FLAGS) $(MAIN)

# Regenerate captured listing outputs from the testing chapter manifest
listings:
	cd .. && uv run python scripts/build_listings.py build testing

# Verify committed listing outputs still match what the code would produce.
# This is what gates the PDF build.
verify-listings:
	cd .. && uv run python scripts/build_listings.py verify testing

# Escape hatch for prose-only iteration: build the PDF without verifying listings.
pdf-no-verify: $(MAIN).tex
	$(LATEXMK) $(LATEXMK_FLAGS) $(MAIN)

# Continuous build — rebuilds on file changes
watch:
	$(LATEXMK) $(LATEXMK_FLAGS) -pvc $(MAIN)

clean:
	$(LATEXMK) -c

distclean:
	$(LATEXMK) -C

.PHONY: all watch clean distclean listings verify-listings pdf-no-verify
