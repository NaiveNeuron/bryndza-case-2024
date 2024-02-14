from functools import cache
import chromadb
import typer
from pathlib import Path
from flashrank import Ranker, RerankRequest


def main(query: str, index_name: str, index_path: Path, rerank: bool = False):
    client = chromadb.PersistentClient(path=str(index_path))
    collection = client.get_collection(name=index_name)

    result = collection.query(query_texts=[query], n_results=10)

    results = [
        {
            "id": id,
            "text": text,
            "meta": metadata
        } for id, text, metadata in zip(*result['ids'], *result['documents'], *result['metadatas'])
    ]


    if rerank:
        ranker = Ranker(model_name="rank-T5-flan", cache_dir="cache")
        rerank_request = RerankRequest(
            query=query,
            passages=results,
        )
        results = ranker.rerank(rerank_request)

    for result in results[:5]:
        text = result["text"]
        metadata = result["meta"]
        print(f"Input tweet: {text}\n\nPrediction: {metadata['label']}\n\n")


if __name__ == "__main__":
    typer.run(main)
