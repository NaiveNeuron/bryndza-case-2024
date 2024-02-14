import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from sklearn.metrics import f1_score
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions

random.seed(42)

SYSTEM_PROMPT = """
Analyze the following tweet and classify who the target of the hate speech is. Use the identified patterns and specific examples from the training data for classification. The categories are:

## Categories

1. Individual - Involves direct attacks on specific individuals. Common examples include derogatory remarks about individuals like "Trump" or "Greta Thunberg". Look for usage of individual names and personal attacks.

2. Organization - Involves criticisms targeted at larger entities such as governments, companies, or specific organizations. Key examples include attacks on 'Government', 'Big oil companies', 'Australia' (referring to its government), 'Wilderness Committee', and the 'EU'. Look for mentions of these entities and critiques of their policies or actions.

3. Community - Involves attacks on broader communities or societal groups. Typical terms used include 'White, middle class, educated, low earners', 'humans', 'adult society', and 'politicians'. This category shifts the focus from a single party to collective human behavior, demographic groups, or societal constructs.

Use chain of thought reasoning to explain your classification. Return a response only for the tweet in the Prediction section. After analyzing the tweet, classify it as "Prediction: 1" for an individual, "Prediction: 2" for an organization, or "Prediction: 3" for a community. Pick only one option and put it on a new line. If the tweet is a factual statement, classify its target as described above.

## Examples
"""

model = "llama2"

def generate_completion(prompt) -> dict:
    session = requests.Session()

    r = session.post('http://localhost:11434/api/generate', 
              json={
                  "model": model,
                  "prompt": prompt,
                  "stream": False,
                  "options": {
                      "temperature": 0.0,
                      }
                  }
              )
    r.raise_for_status()
    return r.json()


def predict(system_prompt, user_prompt) -> str:
    completion = generate_completion(f"{system_prompt}\n\n### Prediction\n\nInput tweet: {user_prompt}")

    return completion["response"]

def parse_prediction(completion: str) -> int:
    individual_answer_options = [
        "Prediction: 1",
        "'Prediction: 1'",
        "'Prediction: 1'.",
        "Prediction: 1 (Individual)"
    ]

    organization_answer_options = [
        "Prediction: 2",
        "'Prediction: 2'",
        "'Prediction: 2'.",
        "Prediction: 2 (Organization)"
    ]

    community_answer_options = [
        "Prediction: 3",
        "'Prediction: 3'",
        "'Prediction: 3'.",
    ]

    if "Prediction: 1" in completion or "individual" in completion.lower():
        return 1
    if "Prediction: 2" in completion or "organization" in completion.lower():
        return 2
    if "Prediction: 3" in completion or "community" in completion.lower():
        return 3


    if any(completion.endswith(option) or completion.startswith(option) for option in individual_answer_options):
        return 1
    elif any(completion.endswith(option) or completion.startswith(option) for option in organization_answer_options):
        return 2
    elif any(completion.endswith(option) or completion.startswith(option) for option in community_answer_options):
        return 3
    else:  # TODO: Failed to parse, raise an error instead
        print(f"Failed to parse prediction: {completion}")
        return False


def classify_example(system_prompt, user_prompt) -> bool:
    index_name = "subtask_b_index_all-mpnet-base-v2"
    index_path = "indexes/subtask_b_index_all-mpnet-base-v2"

    # index_name = "subtask_b_index"
    # index_path = "indexes/subtask_b_index"

    client = chromadb.PersistentClient(path=str(index_path))
    collection = client.get_collection(name=index_name)
    # results = collection.query(query_texts=[user_prompt], n_results=6)

    embedding_model = "all-mpnet-base-v2"
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    results = collection.query(query_embeddings=embedding_fn([user_prompt]), n_results=6)

    examples = ""
    for text, metadata in zip(*results['documents'], *results['metadatas']):
        examples += f"Input tweet: {text}\n\nPrediction: {metadata['label']}\n\n"

    completion = predict(system_prompt + "\n" + examples, user_prompt)
    return parse_prediction(completion), completion


def classify_test_set_parallel(filename, system_prompt, output_filename):
    file = pd.read_csv(filename, index_col=0)
    results = []

    for index, row in tqdm(file.iterrows(), total=len(file)):
        try:
            prediction, completion = classify_example(system_prompt, row["tweet"])
            results.append(
                {"index": index, "prediction": prediction, "completion": completion}
            )
        except Exception as exc:
            print(f"Tweet at index {index} generated an exception: {exc}")

    results = sorted(results, key=lambda x: x["index"])


    with open(output_filename, "w") as outfile:
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
        "SubTask-B(index,tweet)test.csv", SYSTEM_PROMPT,
        "ollama_test_set_predictions_b.jsonl"
    )  # Generates json ready for submission
    # classify_test_set_parallel("SubTask-A-train.csv", SYSTEM_PROMPT)

    # f1 = calculate_f1_score('test_set_predictions.json', 'SubTask-A-train.csv')
    # print(f"F1 Score: {f1}")
