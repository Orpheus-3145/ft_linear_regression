import pandas as pd

def load_csv(csv_path: str):
    df: pd.DataFrame = pd.read_csv(csv_path, delimiter=",", header=0)

    for row in df.itertuples(index=False):
        print(f"km: {row.km} -> {row.price}")