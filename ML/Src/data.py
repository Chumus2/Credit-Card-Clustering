import pandas as pd
from pathlib import Path



def load_csv_df(
    df_path: str | Path,
) -> pd.DataFrame:
    """Load a csv file into a dataframe with error handling.

    Firstly finding file by df_path argument (pathlib.Path).
    Then trying read this dataframe.

    1. if dataframe found:
        - printing success message
        - printing dataframe stats (shape and memory usage)
        - return dataframe
    2. otherwise:
        - printing failure message
        - raises error

    function required arguments:
    1. df_path (str / pathlib.Path) file path to dataframe:

    function returns:
    1. df (pd.DataFrame) founded dataframe by df_path:
    """

    df_path = Path(df_path)

    if not df_path.is_file():
        raise FileNotFoundError(f"File not found: {df_path}")

    try:
        df = pd.read_csv(df_path)
        print(f"Successfully loaded dataframe: {df_path}")

    except Exception as e:
        print(f"Failed to load dataframe: {df_path}")
        raise e

    df_memory_usage = df.memory_usage(deep=True).sum() / 1024 ** 2

    print(f"Dataframe has: {df.shape[0]} rows / {df.shape[1]} columns")
    print(f"Dataframe size: {df_memory_usage:.3f}MB")

    return df


def save_csv_df(
    df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save a csv file to path specified by output_path argument.

    Firstly finding place to save dataframe by output_path argument (pathlib.Path).
    Then trying save dataframe to this path.

    1. if path founded:
        - printing success message
        - printing dataframe stats (shape and memory usage)
        - saves dataframe to it
    2. otherwise:
        - printing failure message
        - raises error

    function required arguments:
    1. df (pd.DataFrame) dataframe to save:
    2. output_path (str / pathlib.Path) path to save dataframe:

    No return.
    """

    output_path = Path(output_path)

    if not output_path.parent.exists():
        raise FileNotFoundError(f"Failed to save file: {output_path}")

    try:
        df.to_csv(output_path, index=False)
        print(f"Successfully saved dataframe: {output_path}")
    except Exception as e:
        print(f"Failed to save dataframe: {output_path}")
        raise e

    df_memory_usage = df.memory_usage(deep=True).sum() / 1024 ** 2

    print(f"Dataframe has: {df.shape[0]} rows / {df.shape[1]} columns")
    print(f"Dataframe memory usage: {df_memory_usage:.3f}MB")