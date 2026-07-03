from tex2qmd.project import parse_chapter_order, qmd_name, render_quarto_yml


def test_parse_chapter_order():
    """Include targets are returned in document order, comments ignored."""
    book_tex = (
        "\\begin{document}\n"
        "\\include{preface}\n"
        "% \\include{skipped}\n"
        "\\include{book-introduction}\n"
        "\\include{book-testing}\n"
    )
    assert parse_chapter_order(book_tex) == [
        "preface",
        "book-introduction",
        "book-testing",
    ]


def test_qmd_name():
    """An include target maps to a .qmd filename."""
    assert qmd_name("book-testing") == "book-testing.qmd"


def test_render_quarto_yml_contains_chapters_and_bib():
    """The rendered config lists chapters and wires up bibliography + csl."""
    yml = render_quarto_yml(
        ["index.qmd", "book-testing.qmd"], "Better Code, Better Science", "Russell A. Poldrack"
    )
    assert "type: book" in yml
    assert "index.qmd" in yml
    assert "book-testing.qmd" in yml
    assert "bibliography: references.bib" in yml
    assert "csl: cambridge.csl" in yml
