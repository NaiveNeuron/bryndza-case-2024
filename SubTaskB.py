import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import AzureOpenAI
from sklearn.metrics import f1_score
from tqdm import tqdm

random.seed(42)

SYSTEM_PROMPT = """
Analyze the following tweet and classify who the target of the hate speech is. Use the identified patterns and specific examples from the training data for classification. The categories are:

## Categories

1. Individual - Involves direct attacks on specific individuals. Common examples include derogatory remarks about individuals like "Trump" or "Greta Thunberg". Look for usage of individual names and personal attacks.

2. Organization - Involves criticisms targeted at larger entities such as governments, companies, or specific organizations. Key examples include attacks on 'Government', 'Big oil companies', 'Australia' (referring to its government), 'Wilderness Committee', and the 'EU'. Look for mentions of these entities and critiques of their policies or actions.

3. Community - Involves attacks on broader communities or societal groups. Typical terms used include 'White, middle class, educated, low earners', 'humans', 'adult society', and 'politicians'. This category shifts the focus from a single party to collective human behavior, demographic groups, or societal constructs.

Use chain of thought reasoning to explain your classification. After analyzing the tweet, classify it as "Prediction: 1" for an individual, "Prediction: 2" for an organization, or "Prediction: 3" for a community. Pick only one option and put it on a new line.

## Examples

Input tweet: #GlobalWarming #FridaysForFuture   #ClimateChange #Greenwashing #Renewables  #ClimateStrike  #ExtinctionRebellion #ClimateCrisis  #ClimateAction        

You've been fooled by Greta Thunberg:

Prediction: 1

Input tweet: So now the Palm Oil Industry is trying to rebrand deforestation as ""reforestation""..

I can't describe how angry this makes me!

When will humanity learn that advertising doesn't fix the crisis we are in!?!

#ExtinctionRebellion #TalkingClimate @Lets_Discuss_CC #FridaysForFuture

Prediction: 2

Input tweet: #FridaysforFuture question:

The global north is engaged in worldwide brain-drain of global south - good thing or bad?

Prediction: 3

Input tweet: #FridaysForFuture #ExtinctionRebellion 
How Naomi Klein, Josh Fox, 
George Monbiot, Ketan Joshi,  and many more lie to us

Every day. On  #ClimateChange &amp; #GlobalWarming!

It's #Greenwashing &amp; these people keep lying to us about real green technology emissions, #ClimateAction:

Prediction: 1

Input tweet: +++ GLOBAL CLIMATE STRIKE +++
#FridaysForFuture
The Global South is being ransacked by the Global North, citizens are being exploited by energy giants and the atmosphere is colonised in the name of growth.

We need to uproot this system NOW!! 🔥

#LossAndDamage 
#ClimateJustice https://t.co/EFVVxnTow1

Prediction: 2

Input tweet: To all the climate activists out there who aren't vegan.

You are denying the fact that animal agriculture is an ecological disaster.

If you want change, you have to be the change yourself.

Please go vegan, now!

#vegan #govegan 
#climatechange 
#kiimakrise
#fridaysforfuture https://t.co/in4MvsZA7E

Prediction: 3

Input tweet: we have to wage war against fossil fuel companies because they are earning from the destruction of our planet  @GretaThunberg @vanessa_vash #climatejusticenow #ClimateAction #environmentalprotection #FridaysForFuture #ClimateEmergency #UprootTheSystem https://t.co/8YPuhUXcgm

Prediction: 2

Input tweet: These youth put us all to shame #FridaysForFuture

Prediction: 3
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
                temperature=0,
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
    completion = predict(system_prompt, user_prompt)
    return parse_prediction(completion)


def classify_test_set_parallel(filename, system_prompt, output_filename):
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
                    {"index": original_index, "prediction": prediction}
                )
            except Exception as exc:
                print(f"Tweet at index {original_index} generated an exception: {exc}")

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
        "test_set_predictions_b.jsonl"
    )  # Generates json ready for submission
    # classify_test_set_parallel("SubTask-A-train.csv", SYSTEM_PROMPT)

    # f1 = calculate_f1_score('test_set_predictions.json', 'SubTask-A-train.csv')
    # print(f"F1 Score: {f1}")
