from tex2qmd.project import (
    load_index_body,
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
    # each chapter's end-of-page references get a heading
    assert "reference-section-title:" in yml
    assert "References" in yml


def test_render_quarto_yml_adds_navbar_tools_when_repo_url_given():
    """A repo_url yields navbar tools linking to source code and issue reporting."""
    yml = render_quarto_yml(
        ["index.qmd"],
        "Better Code, Better Science",
        "Russell A. Poldrack",
        repo_url="https://github.com/BetterCodeBetterScience/bettercode",
    )
    assert "navbar:" in yml
    # GitHub source-code tool
    assert "icon: github" in yml
    assert "https://github.com/BetterCodeBetterScience/bettercode" in yml
    # Report-an-issue tool
    assert "Report an issue" in yml
    assert "https://github.com/BetterCodeBetterScience/bettercode/issues/new" in yml


def test_render_quarto_yml_omits_navbar_without_repo_url():
    """No repo_url means no navbar block is emitted."""
    yml = render_quarto_yml(
        ["index.qmd"], "Better Code, Better Science", "Russell A. Poldrack"
    )
    assert "navbar:" not in yml


def test_render_quarto_yml_includes_subtitle_when_given():
    """A non-empty subtitle is emitted as a book subtitle line."""
    yml = render_quarto_yml(
        ["index.qmd"],
        "Better Code, Better Science",
        "Russell A. Poldrack",
        subtitle="Software Engineering for Reproducible Science in the Age of AI",
    )
    assert "subtitle: \"Software Engineering for Reproducible Science in the Age of AI\"" in yml


def test_render_cover_contains_front_matter_and_body():
    """The cover carries title/subtitle/author front matter and the given body verbatim."""
    body = "Hello reader.\n\nEnjoy the book."
    cover = render_cover(
        "Better Code, Better Science",
        "Software Engineering for Reproducible Science in the Age of AI",
        "Russell A. Poldrack",
        body,
    )
    assert 'title: "Better Code, Better Science"' in cover
    assert 'subtitle: "Software Engineering for Reproducible Science in the Age of AI"' in cover
    assert 'author: "Russell A. Poldrack"' in cover
    assert body in cover


def test_load_index_body_reads_given_file(tmp_path):
    """load_index_body returns the contents of the given file."""
    body_file = tmp_path / "index_body.md"
    body_file.write_text("Custom landing text.\n")
    assert load_index_body(body_file) == "Custom landing text.\n"


def test_load_index_body_default_file_exists():
    """The default landing-body file ships with the pipeline and is non-empty."""
    body = load_index_body()
    assert body.strip()
