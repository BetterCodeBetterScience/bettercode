clean:
	- rm -rf book/_build

build-html: clean
	myst build --html
	npx serve _build/html

build-pdf:
	jupyter-book build book/ --builder pdflatex

check-links:
	check-links	

pipinstall:
	uv pip install -r pyproject.toml
	uv pip install -e .

test-textmining:
	py.test --cov=src/codingforscience/textmining  --cov-report term-missing -v src/codingforscience/textmining

test-property:
	py.test -v --hypothesis-show-statistics src/codingforscience/property_based_testing

test-simple:
	py.test src/codingforscience/simple_testing

docker-build:
	docker build -t condatest .

docker-shell:
	docker run -it -w /root --entrypoint=bash condatest

# --- Quarto HTML edition (GitHub Pages) ---------------------------------
# The book prose lives in a SEPARATE PRIVATE repo, cloned into ./latex locally.
# This repo holds the public code examples, the tex2qmd pipeline, and the
# rendered site under docs/ (served by Pages from `main` /docs).
# `make site` needs ./latex present plus pandoc, quarto, and pdftocairo on PATH.

site:
	cd latex && $(MAKE) verify-listings
	uv run python scripts/tex2qmd/convert.py --latex-dir latex --out web
	cd web && quarto render
	touch docs/.nojekyll   # stop GitHub Pages' Jekyll from mangling Quarto assets

site-preview:
	uv run python scripts/tex2qmd/convert.py --latex-dir latex --out web
	cd web && quarto preview

.PHONY: site site-preview



