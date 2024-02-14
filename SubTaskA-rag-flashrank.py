import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import AzureOpenAI
from sklearn.metrics import f1_score
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from flashrank import Ranker, RerankRequest

random.seed(42)

SYSTEM_PROMPT = """
Analyze the input tweet to determine if it is hate speech or not, based on the following criteria:

## Hate Speech Patterns

1. Presence of "You've been fooled by Greta Thunberg" or #FridaysForFuture in the tweet.
2. Embodies aggression or contempt towards specific groups or institutions, including dismissive attitudes towards climate activists, criticism of world leaders for climate inaction, or strong sentiments against companies investing in fossil fuels.
3. Frequent use of negative language, such as 'shame', 'lie', 'greedy', 'fake', 'idiot', to express dissatisfaction or attack others.
4. Highlights a strong ideological alignment or belief, often against fossil fuels and blaming capitalism for the climate crisis, indicating belief-driven intolerance.
5. The tone is accusatory, confrontational, and not oriented towards dialogue or understanding.

## Non-Hate Speech Patterns

1. Expresses concern about climate change and promotes action without aggression or contempt. Advocates for policy changes, shares environmental information, and encourages collective action rather than targeting individuals or groups.
2. Lacks negative language or personal attacks.
3. Presents a clear ideological stance on climate change in a constructive or informative manner, aiming to educate or raise awareness rather than cast blame.
4. The tone is conversational and informative, promoting understanding and engagement rather than confrontation.

## Evaluation

- If the tweet aligns more with the Hate Speech Patterns, output: 'Prediction: 1' (indicating it is hate speech).
- If the tweet aligns more with the Non-Hate Speech Patterns, output: 'Prediction: 0' (indicating it is not hate speech).

## Examples
"""

api_version = "2023-07-01-preview"
endpoint = os.getenv("OPENAI_BASE_URL")

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=os.getenv("OPENAI_API_KEY"),
)


def predict(system_prompt, user_prompt) -> str:
    for attempt in range(25):
        try:
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Input tweet: {user_prompt}"},
                ],
                temperature=0.1,
            )

            return completion.choices[0].message.content

        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")

            if "rate limit" in str(e).lower():
                time_to_wait = random.randint(5, 15)
                print(
                    f"Rate limited. Waiting for {time_to_wait} seconds before retrying..."
                )
                time.sleep(time_to_wait)
            elif "Azure OpenAI's content management policy" in str(e):
                print("ContentFilterError: returning prediction 1")
                return "Prediction: 1 (indicating it is hate speech)."
            else:
                time.sleep(random.randint(5, 15))

    return "Prediction failed after multiple attempts."


def parse_prediction(completion: str) -> bool:
    hate_answer_options = [
        "Prediction: 1",
        "'Prediction: 1'",
        "'Prediction: 1'.",
        "Prediction: 1 (indicating it is hate speech).",
        "'Prediction: 1' (indicating it is hate speech).",
    ]

    non_hate_answer_options = [
        "Prediction: 0",
        "'Prediction: 0'",
        "'Prediction: 0'.",
        "Prediction: 0 (indicating it is not hate speech).",
        "'Prediction: 0' (indicating it is not hate speech).",
    ]

    if any(completion.endswith(option) for option in hate_answer_options):
        return True
    elif any(completion.endswith(option) for option in non_hate_answer_options):
        return False
    else:  # TODO: Failed to parse, raise an error instead
        print(f"Failed to parse prediction: {completion}")
        return False


def classify_example(system_prompt, user_prompt) -> bool:
    # index_name = "subtask_a_index"
    # index_path = "indexes/subtask_a_index"

    # client = chromadb.PersistentClient(path=str(index_path))
    # collection = client.get_collection(name=index_name)

    # result = collection.query(query_texts=[user_prompt], n_results=10)

    index_name = "subtask_a_index_all-mpnet-base-v2"
    index_path = "indexes/subtask_a_index_all-mpnet-base-v2"
    embedding_model = "all-mpnet-base-v2"

    client = chromadb.PersistentClient(path=str(index_path))
    collection = client.get_collection(name=index_name)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    result = collection.query(query_embeddings=embedding_fn([user_prompt]), n_results=12)

    results = [
        {
            "id": id,
            "text": text,
            "meta": metadata
        } for id, text, metadata in zip(*result['ids'], *result['documents'], *result['metadatas'])
    ]

    ranker = Ranker(model_name="rank-T5-flan", cache_dir="cache")
    rerank_request = RerankRequest(
        query=user_prompt,
        passages=results,
    )
    results = ranker.rerank(rerank_request)

    examples = ""
    for result in results[:6]:
        text = result["text"]
        metadata = result["meta"]
        examples += f"Input tweet: {text}\n\nPrediction: {metadata['label']}\n\n"

    completion = predict(system_prompt + "\n" + examples, user_prompt)
    return parse_prediction(completion)


def classify_test_set_parallel(filename, system_prompt):
    file = pd.read_csv(filename, index_col=0)
    results = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_tweet = {
            executor.submit(classify_example, system_prompt, row["tweet"]): row.name
            for index, row in file.iterrows()
        }
        for future in tqdm(as_completed(future_to_tweet), total=len(file)):
            original_index = future_to_tweet[future]
            try:
                prediction = future.result()
                results.append(
                    {"index": original_index, "prediction": 1 if prediction else 0}
                )
            except Exception as exc:
                print(f"Tweet at index {original_index} generated an exception: {exc}")

    results = sorted(results, key=lambda x: x["index"])

    with open("test_set_predictions.jsonl", "w") as outfile:
        for result in results:
            outfile.write(json.dumps(result) + "\n")


def read_true_labels(training_set_filename):
    training_set = pd.read_csv(training_set_filename, index_col="index")
    return training_set["label"]


def calculate_f1_score(predictions_filename, training_set_filename):
    true_labels = read_true_labels(training_set_filename)

    with open(predictions_filename, "r") as file:
        predictions = json.load(file)

    predicted_labels = {item["index"]: item["prediction"] for item in predictions}

    aligned_predictions = [predicted_labels.get(idx, 0) for idx in true_labels.index]

    return f1_score(true_labels, aligned_predictions)


def find_patterns_in_dataset():
    file = pd.read_csv("SubTask-A-train.csv")
    hate_speech_texts = set(file[file["label"] == 1]["tweet"])
    non_hate_speech_texts = set(file[file["label"] == 0]["tweet"])

    random.shuffle(list(hate_speech_texts))
    random.shuffle(list(non_hate_speech_texts))

    n_examples = 30

    # hate_speech_texts_without_greta = [text for text in hate_speech_texts if "You've been fooled by Greta" not in text]

    hate_speech_texts_without_greta_formatted = ""
    for idx, text in enumerate(list(hate_speech_texts)[:n_examples]):
        hate_speech_texts_without_greta_formatted += f"{idx + 1}. {text}\n---\n"

    non_hate_speech_texts_formatted = ""
    for idx, text in enumerate(list(non_hate_speech_texts)[:n_examples]):
        non_hate_speech_texts_formatted += f"{idx + 1}. {text}\n---\n"

    all_texts_formatted = ""
    all_texts_formatted += (
        f"\n\n>>>> Hate speech:\n{hate_speech_texts_without_greta_formatted}\n---\n"
    )
    all_texts_formatted += (
        f"\n\n>>>> Non-hate speech:\n{non_hate_speech_texts_formatted}\n---\n"
    )

    system_prompt = f"""You will be given {n_examples} tweets that were classified as hate speech. Your task is to find a
    common " "pattern these texts share and figure out why they were classified as hate speech. For a good "
    "comparison, I will also send you {n_examples} non-hate speech tweets so you have something to compare it to. 
    Since these are tweets, focus on hashtags (#)."""

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": all_texts_formatted},
        ],
    )

    return completion.choices[0].message.content


if __name__ == "__main__":
    # reasoning = find_patterns_in_dataset()
    # print(reasoning)

    classify_test_set_parallel(
        "SubTask-A-(index,tweet)test.csv", SYSTEM_PROMPT
    )  # Generates json ready for submission
    # classify_test_set_parallel("SubTask-A-train.csv", SYSTEM_PROMPT)

    # f1 = calculate_f1_score('test_set_predictions.json', 'SubTask-A-train.csv')
    # print(f"F1 Score: {f1}")
