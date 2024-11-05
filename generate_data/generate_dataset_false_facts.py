from openai import OpenAI
import os
from datasets import load_dataset
import json

dataset_name = "hotpotqa" # hotpotqa, triviaqa, or nq

API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
client = OpenAI(
  organization='org-5O1cuTOV2xUKmJWAOW7YxnfU',
  api_key = os.environ['OPENAI_KEY']
)

def generate_chat_completion(messages, model="gpt-4o-mini", max_tokens=None, log_probs=True):
    results = client.chat.completions.create(
        model=model,
        messages=messages,
        logprobs=log_probs
    )
    return results.choices[0].message.content

out_file = f"{dataset_name}_1000.json"
tf_data = load_dataset("json", data_files=f"{dataset_name}_1000_initial.json")
data_iter = iter(tf_data["train"])

with open(out_file, 'a') as f:
    f.write('[')

current_id = 0
for i in range(len(tf_data["train"])):
    datapoint = next(data_iter)
    c = datapoint["context"]
    a = datapoint["answer"]
    q = datapoint["question"]

    input_str = f"Look at the following entity: {a}. First, think about what type of entity it is, e.g. a name, a place, a date, or similar. Then, come up with a different example of the same entity. For example, if it was name, return a different name. If it was a place, return a different place, etc. Here is the entity: {a}. Return the new example only, don't say anything else."
    message_dict = [{
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": input_str
      }]

    new_entity = generate_chat_completion(message_dict, model="gpt-4o-mini", max_tokens=None, log_probs=True).replace(".", "").replace("New example: ", "")

    if a in c:
        context_sentence = c.replace(a, new_entity)
    elif a.lower() in c:
        context_sentence = c.replace(a.lower(), new_entity.lower())

    entry = {
        "id": current_id,
        "question": q,
        "true_answer": a,
        "wrong_answer": new_entity,
        "context": context_sentence
    }
    current_id += 1
    
    if current_id % 100 == 0:
        print(f"Current sample ID: {current_id}")
 
    with open(out_file, 'a') as f:
        f.write(json.dumps(entry, indent=4))
        f.write(',\n')

with open(out_file, 'a') as f:
    f.write(']')
