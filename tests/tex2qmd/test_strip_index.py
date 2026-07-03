from tex2qmd.preprocess import strip_index


def test_strip_simple_index():
    """A trailing \\index command is removed, prose preserved."""
    tex = "Tests matter.\\index{testing} They do."
    assert strip_index(tex) == "Tests matter. They do."


def test_strip_subentry_index():
    """Subentry index (with !) is removed."""
    tex = "unit tests\\index{testing!unit} are good."
    assert strip_index(tex) == "unit tests are good."


def test_strip_multiple_index():
    """Multiple index commands on one line are all removed."""
    tex = "a\\index{x}b\\index{y}c"
    assert strip_index(tex) == "abc"
