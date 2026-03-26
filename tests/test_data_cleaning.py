import pandas as pd


def test_placeholder():
    # Keep test suite non-empty; expand later when you expose cleaning helpers.
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert df.shape[0] == 3
