import importlib


def test_backtesting_module_imports():
    m = importlib.import_module("var_project.risk.backtesting")
    assert m is not None
