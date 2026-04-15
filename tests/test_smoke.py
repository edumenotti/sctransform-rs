import sctransform_rs


def test_version_is_string():
    assert isinstance(sctransform_rs.__version__, str)
    assert sctransform_rs.__version__ == "0.1.0"


def test_add_smoke():
    assert sctransform_rs.add(2, 3) == 5
    assert sctransform_rs.add(-1, 1) == 0
    assert sctransform_rs.add(0, 0) == 0
