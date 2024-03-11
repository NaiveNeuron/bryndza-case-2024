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

Input tweet: This is the future we are leaving behind for our children. World leaders must take #ClimateAction now! 

Climate change is causing 'super-extreme' weather events, scientists say https://t.co/WBtjFNrNPP via @YahooNews @Fridays4future #FridaysForFuture

Prediction: 0

Input tweet: #ExtinctionRebellion #ClimateAction #ClimateCrisis #GlobalWarming #ClimateChange #FridaysForFuture #ClimateStrike #Renewables #Greenwashing     

You've been fooled by Greta Thunberg:

Prediction: 1

Input tweet: @GretaThunberg The tragedy is that most of us will become climate activists sooner or later.

Please, please make it sooner.

DEMAND meaningful #ClimateActionNow 

#PeopleNotProfit #FridaysForFuture #ClimateStrike #COP27 https://t.co/0616AmD6rR

Prediction: 0

Input tweet: @GeraldKutney Good lord what a effed up thread you have on this post.
Patrick Moore enthusiasts. 
#ClimateChange #ClimateCrisis #ClimateBrawl #ClimateStrike #Climatepoli #FridaysForFuture  @ExtinctionR 
https://t.co/FCLWQY6TF0

Prediction: 1

Input tweet: Week82 #ClimateStrike🌍 in #India🇮🇳
⚠️ #PeopleNotProfit ⚠️

⌛️We are running out of time! Without ambitious action on #ClimateChange by 2022, We'll pass the tipping points for irreversible damage to our #Planet.

#FridaysForFuture
@GretaThunberg @vanessa_vash @ExtinctionR @UNFCCC https://t.co/3A8NyzyHby

Prediction: 0

Input tweet: @TheFigen_ Brilliant! And to think some sick old racist people would nuke this World to oblivion! Mother Nature &amp; Democracy &amp; Freedom, Liberty and Human Rights needs defending! #FridaysForFuture #GretaThunbergIsCorrect #1SOS

Prediction: 1

Input tweet: Fires can be deadly,destroying homes,#wildlife habitat,timbers &amp; polluting the air with emissions harmful to human health #wildfire increases #AirPollution @UNFCCC @Fridays4future #UprootTheSystem #ClimateEmergency #ClimateActionNow #FridaysForFuture #biodiversity Climate Change https://t.co/TT4T63qA2s

Prediction: 0

Input tweet: This is why UK politicians are so reluctant to divest from fossil fuels:   1/7

@GOVUK #Corruption #ToryCorruption #ExtinctionRebelliom #XR #KeepItInTheGround #ClimateJustice #FridaysForFuture #GreenNewDeal #UKPolitics #TalkingClimate @Lets_Discuss_CC

Prediction: 1

Input tweet: #GlobalWarming #FridaysForFuture #ClimateChange #Greenwashing   #Renewables  #ClimateStrike  #ExtinctionRebellion #ClimateAction  #ClimateCrisis

Solar + batteries = 265g CO2 kWh
450g =soil rich with CO2 destroyed (installation) +Toxic waste dump from panels production. Thread:

Prediction: 0

Input tweet: #ExtinctionRebellion #FridaysForFuture 
How Naomi Klein, Josh Fox, 
George Monbiot, Ketan Joshi, and many more lie to us

Every day. On #GlobalWarming &amp; #ClimateChange!

It's #Greenwashing &amp; these people keep lying to us about real green technology emissions, #ClimateAction:

Prediction: 1
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
    completion = predict(system_prompt, user_prompt)
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
