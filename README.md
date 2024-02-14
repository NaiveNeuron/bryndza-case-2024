# Bryndza @ CASE 2024

This repository contains the code that has been used to prepare the submissions
of the team "Bryndza" to the Shared task on Climate Activism Stance and Hate
Event Detection at [CASE 2024](https://emw.ku.edu.tr/case-2024/).

<p align="center">
  <img src="https://github.com/NaiveNeuron/bryndza-case-2024/blob/main/images/graph.png?raw=true" />
</p>


## Installation

The dependencies are managed using [poetry](https://python-poetry.org/).

To install them, run the following:

    $ poetry install

## Dataset Preparation

As some of the datasets for the shared tasks were distributed separately as
tweet and label files, these need to be merged into a single `.csv` file.

To do so, the script `utils/csv_merge.py` can be used, assuming the raw files
can be found in the `raw_dataset` directory. The script can be executed as
follows:

    $ poetry run python utils/csv_merge.py raw_dataset/SubTask-A-\(index,tweet\)val.csv raw_dataset/SubTask-A\(index,label\)val.csv index dataset/SubTask-A-val.csv

    $ poetry run python utils/csv_merge.py raw_dataset/SubTask-B\(index,tweet\)val.csv raw_dataset/SubTask-B\(index,label\)val.csv index dataset/SubTask-B-val.csv

    $ poetry run python utils/csv_merge.py raw_dataset/SubTask-C\(index,tweet\)val.csv raw_dataset/SubTask-C\(index,label\)val.csv index dataset/SubTask-C-val.csv

## Evaluation

In order to run the evaluation scripts, the indexes need to be created first.
This can be done by executing a command such as the following:

    $ poetry run python create_index.py SubTask-A-train.csv SubTask-A-val.csv 'subtask_a_index' indexes/subtask_a_index

The script also supports creation of indexes using specific Sentence
Transformers models, for instance:

    $ poetry run python create_index.py SubTask-A-train.csv SubTask-A-val.csv subtask_a_index_all-mpnet-base-v2 indexes/subtask_a_index_all-mpnet-base-v2 --embedding-model all-mpnet-base-v2

The respective evaluation scripts contained in `SubTask*.py` files can then be
run using command similar to the following:

    $ poetry run python SubTaskA.py

Note that in order for them to run correctly the `OPENAI_API_KEY` and
`OPENAI_BASE_URL` need to be set up appropriately, as the code utilizess
`AzureOpenAI` internally.

### Evaluating LLaMA via Ollama

In order to reproduce the LLaMA-based evaluation, it is first necessary to spin
the model up. This has been done using [Ollama](https://ollama.ai/) in our
experiments by running:

    $ ollama run llama2:70b-chat

The `ollama-*.py` scripts in this repository can then be run using commands
similar to the following:

    $ poetry run python ollama-SubTaskB.py

## Model Prediction Error Annotations

As part of the analysis of the errors made by the best performing models, the
instances where the predictions did not agree with the ground truth labels have
been annotated.

These annotations can be found in the `annotations/` folder in the form of
three CSV files. All of these follow the same format:

- `tweet`: the content of the tweet
- `prediction`: the prediction of the model
- `label`: the ground truth label provided in the test dataset
- `class`: the class or type of the issue as annotated by the human annotator -- can be one of the following:
    - `error`: the model made an error
    - `wrong-label`: the human annotator declared that the `label` provided in the golden set was incorrect
    - `unclear`: it was not clear to the human annotator whether it was the model who made a mistake or whether the `label` was wrong

## Cite

If you find any of the tools and/or results useful, we would appreciate if you
could cite the [associated article](https://arxiv.org/abs/2402.06549):

```
@article{vsuppa2024bryndza,
  title={Bryndza at ClimateActivism 2024: Stance, Target and Hate Event Detection via Retrieval-Augmented GPT-4 and LLaMA},
  author={{\v{S}}uppa, Marek and Skala, Daniel and Ja{\v{s}}{\v{s}}, Daniela and Su{\v{c}}{\'\i}k, Samuel and {\v{S}}vec, Andrej and Hra{\v{s}}ka, Peter},
  journal={arXiv preprint arXiv:2402.06549},
  year={2024}
}
```
