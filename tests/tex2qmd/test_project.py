from tex2qmd.project import (
    parse_chapter_order,
    qmd_name,
    render_cover,
    render_quarto_yml,
)


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
    assert "output-dir: ../docs" in yml
    assert "index.qmd" in yml
    assert "book-testing.qmd" in yml
    assert "bibliography: references.bib" in yml
    assert "csl: cambridge.csl" in yml


def test_render_quarto_yml_includes_subtitle_when_given():
    """A non-empty subtitle is emitted as a book subtitle line."""
    yml = render_quarto_yml(
        ["index.qmd"],
        "Better Code, Better Science",
        "Russell A. Poldrack",
        subtitle="Software Engineering for Reproducible Science in the Age of AI",
    )
    assert "subtitle: \"Software Engineering for Reproducible Science in the Age of AI\"" in yml


def test_render_cover_contains_title_subtitle_author():
    """The cover landing page carries title/subtitle/author front matter and a welcome."""
    cover = render_cover(
        "Better Code, Better Science",
        "Software Engineering for Reproducible Science in the Age of AI",
        "Russell A. Poldrack",
    )
    assert 'title: "Better Code, Better Science"' in cover
    assert 'subtitle: "Software Engineering for Reproducible Science in the Age of AI"' in cover
    assert 'author: "Russell A. Poldrack"' in cover
    assert "Welcome to the web edition" in cover
