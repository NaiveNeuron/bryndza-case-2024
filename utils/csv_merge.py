import typer
import pandas as pd

from pathlib import Path

def main(text_file: Path, label_file: Path, merge_column: str, out_file: Path):

    df_text = pd.read_csv(text_file)
    df_label = pd.read_csv(label_file)

    merged_df = pd.merge(df_text, df_label, on=merge_column)

    print(merged_df)
    merged_df.to_csv(out_file, index=False)


if __name__ == "__main__":
    typer.run(main)
