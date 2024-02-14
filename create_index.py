import typer
from pathlib import Path
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

def main(train_data: Path, val_data: Path, index_name: str, index_path: Path, embedding_model: str = "all-MiniLM-L6-v2"):
    df_train = pd.read_csv(train_data)
    df_val = pd.read_csv(val_data)

    df = pd.concat([df_train, df_val], ignore_index=True)

    client = chromadb.PersistentClient(path=str(index_path))
    collection = client.create_collection(name=index_name)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)

    for _, row in tqdm(df.iterrows()):
        collection.add(
            embeddings=embedding_fn([row["tweet"]]),
            documents=[row["tweet"]],
            metadatas=[{"label": row["label"]}],
            ids=[str(row["index"])]
        )


if __name__ == "__main__":
    typer.run(main)
