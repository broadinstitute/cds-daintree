import pandas as pd
from daintree_runner.gather import read_row_concatenated_csvs, read_col_concatenated_csvs

def test_read_row_concatenated_csvs(tmpdir):
    index = 0

    def to_fn(m):
        nonlocal index
        index += 1
        fn = tmpdir.join(f"{index}.csv")
        m.to_csv(fn, index=False)
        return fn

    fn1 = to_fn(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
    fn2 = to_fn(pd.DataFrame({"a": [5, 6], "b": [7, 8]}))

    df = read_row_concatenated_csvs([fn1, fn2])

    assert list(df["a"]) == [1, 2, 5, 6]
    assert list(df["b"]) == [3, 4, 7, 8]

def read_col_concatenated_csvs(tmpdir):
    index = 0

    def to_fn(m):
        nonlocal index
        index += 1
        fn = tmpdir.join(f"{index}.csv")
        m.to_csv(fn)
        return fn

    fn1 = to_fn(pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=["i0", "i1"]))
    fn2 = to_fn(pd.DataFrame({"c": [1, 2], "d": [3, 4]}, index=["i1", "i0"]))

    df = read_row_concatenated_csvs([fn1, fn2])

    assert list(df.loc["i0",["a", "b", "c", "d"]]) == [1,3,2,4]
    assert list(df.loc["i1",["a", "b", "c", "d"]]) == [2,4,1,3]
