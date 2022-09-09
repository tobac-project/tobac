import pytest
import tobac


def test_dummy_function():
    assert 1 == 1


def test_version():
    """Test to make sure that we have a version number included.
    If it's not, this should result in an error.
    """
    assert tobac.__version__ == tobac.__version__
