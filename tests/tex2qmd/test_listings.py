import pytest
from tex2qmd.listings import language_for_style


@pytest.mark.parametrize(
    "style,expected",
    [
        ("Python", "python"),
        ("Pythonshort", "python"),
        ("repl", "python"),
        ("replshort", "python"),
        ("shell", "bash"),
        ("shellshort", "bash"),
        ("transcript", "text"),
        ("transcriptshort", "text"),
        ("R", "r"),
        ("yaml", "yaml"),
        ("toml", "toml"),
        ("json", "json"),
        ("jsonshort", "json"),
        ("dockerfile", "dockerfile"),
        ("makefile", "makefile"),
        ("Rust", "rust"),
        ("", "text"),
        ("NonExistentStyle", "text"),
    ],
)
def test_language_for_style(style, expected):
    """Each listing style maps to the expected fenced-code language."""
    assert language_for_style(style) == expected
