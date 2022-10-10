import pytest
import tobac


def test_dummy_function():
    assert 1 == 1


def test_version():
    """Test to make sure that we have a version number included.
    Also test to make sure that the version number complies with
    semantic versioning guidelines.
    If it's not, this should result in an error.
    """
    import re

    assert type(tobac.__version__) is str
    # Make sure that we are following semantic versioning
    # i.e., our version is of form x.x.x, where x are all
    # integer numbers.
    assert re.match(r"[0-9]+\.[0-9]+\.[0-9]+", tobac.__version__) is not None
