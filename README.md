
## About

This repository contains code for our man-in-the-middle (MitM) attack framework targeting the responses of large language models (LLMs). By perturbing inputs in closed-book, fact-based QA settings, the framework demonstrates the vulnerability of LLMs to incorrect responses. Additionally, we train classifiers to detect potential attacks by analyzing response uncertainty.

## Setup

Our experiments include the models GPT-4o, GPT-4o-mini, LLaMA-2-13B, Mistral-7B and Phi-3.5-mini. For using the GPT models, you will need an OpenAI API key. For LLaMA, Mistral and Phi, you will need a huggingface account.

Before running the code, you need to set the following environment variables:

- ```OPENAI_ORG``` (OpenAI organization token)
- ```OPENAI_KEY``` (OpenAI key)
- ```HF_TOKEN``` (huggingface account token)

## How to run: Attack Experiments

- Run ``generate_data/generate_dataset_initial.py`` with all three datasets. After, run ``generate_data/generate_dataset_false_facts.py``. 

- Run ```qa_eval.py```. This gets us the correctly answered questions for each model and dataset.
  
  Example: ```python3 qa_eval.py -m gpt-4o -d triviaqa```

- Run ```uncertainty.py```. This runs the experiments with different prompt injection attacks. The resulting uncertainties of the answers are also recorded here.
  
  Example: ```python3 uncertainty.py -m gpt-4o -v1```. This will prompt GPT-4o with attack number 1, i.e. alpha. 
  
  **Alternatively,** you can simply run the shell scripts for each model (under ```/shell_scripts```).

- Run ```uncertainty_eval.py```. This gives us the results you can see in Tables 2 and 3.

  Example: ```python3 uncertainty_eval.py -m gpt-4o -v1```. This will evaluate the uncertainty scores we received for querying GPT-4o with the alpha attack.

## How to run: Attack Classifiers

- First, prepare the data to train the classifiers on. For this, run ```classifiers/build_data.py```.

- To train the classifiers: run ```classifiers/extratrees_classifier.py```. This will train all four classifiers to distinguish between attacked and unattacked responses, and save the results into ```classifiers/best_extra_trees_results.json```.

- To plot the results, run ```classifiers/aucroc_plot.py``` and ```classifiers/confusion_matrix.py```.