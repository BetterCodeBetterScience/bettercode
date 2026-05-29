This is the subdirectory for the latex build of the book.  the main latex file is `book.tex`.  See the `Makefile` for instructions on how to build it.

## NON-NEGOTIABLE: do not edit narrative text

**NONE of the narrative (prose) text in the `.tex` files may be modified by the
LLM.** Only code inside `lstlisting` blocks and `\lstinputlisting` options
(e.g. `firstline`/`lastline`) may be changed. All section text, headings, captions,
and sentences are the author's and must be left untouched. If a code change makes
surrounding prose inaccurate, surface it to the author — do not edit the prose.

The specific design features for this project are specified within `designchanges.sty`.

For this development project we will focus on only one chapter: `book-testing.tex`.  This is the only chapter that is currently included in the main `book.tex` file.

## Code listings

Code listings are embedded within `lstlisting` blocks.  For most of the chapters (including the testing chapter that we will focus on here), these have been marked with styles referring to the specific content type (the styles are specified within the design file). 