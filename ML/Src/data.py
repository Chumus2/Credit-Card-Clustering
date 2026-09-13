import pandas as pd
from pathlib import Path



def load_csv_df(
    df_path: str | Path,
) -> pd.DataFrame:
    """Load a csv file into a dataframe with error handling

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

    df_size = (df_path.stat().st_size) / (1024 * 1024)

    print(f"Dataframe have: {df.shape[0]} rows / {df.shape[1]} columns")
    print(f"Dataframe size: {df_size:.3f}MB")

    return df