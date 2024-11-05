from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

device = "cuda"
num_samples = 10000 # in the end, the sampling will cut at 1000 - but some samples will be filtered out, so we set the threshold higher
dataset_name = "hotpotqa" # triviaqa, hotpotqa, or nq
out_file = f"../data/{dataset_name}_1000.json"

model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct").to(device)
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if dataset_name == "hotpotqa":
    dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="train", streaming=True, trust_remote_code=True)
elif dataset_name == "triviaqa":
    dataset = load_dataset("mandarjoshi/trivia_qa", "rc", split="train", streaming=True)
elif dataset_name == "nq":
    dataset = load_dataset("google-research-datasets/natural_questions", split="train", streaming=True)
else:
    print("Not a supported dataset.")
    exit()

data_iter = iter(dataset)

with open(out_file, 'a') as f:
    f.write('[')

current_id = 0

for i in range(num_samples):
    datapoint = next(data_iter)

    if dataset_name == "hotpotqa":
        q = datapoint["question"]
        a = datapoint["answer"]
    elif dataset_name == "triviaqa":
        q = datapoint["question"]
        a = datapoint["answer"]["value"]
    elif dataset_name == "nq":
        q = datapoint["question"]["text"]+"?"
        a = datapoint["annotations"]["short_answers"][0]["text"]
        if len(a) == 0:
            continue
        a = a[0]

    query = f"Look at the following question-answer pair: Question: {q}. Answer: {a}. Respond with a factual sentence which shortly states the answer to the question, including all relevant context. Don't add more information than necessary. Respond in one sentence."

    inputs = tokenizer.encode(query, return_tensors="pt").to(device)
    outputs = model.generate(inputs, pad_token_id=tokenizer.pad_token_id, max_new_tokens=35, temperature=0.1, do_sample=True)
    # context = sentence containing the answer to the question
    context = tokenizer.batch_decode(outputs , skip_special_tokens=True)[0]
    context_sentence = context[len(query):]

    if a.lower() in context_sentence.lower():
        context_sentence = context_sentence.replace("Factual sentence: ", "")
        context_sentence = context_sentence.replace("\n", "")
        
        entry = {
            "id": current_id,
            "question": q,
            "answer": a,
            "context": context_sentence
        }
        current_id += 1

        with open(out_file, 'a') as f:
            f.write(json.dumps(entry, indent=4))
            f.write(',\n')
    
    if current_id % 100 == 0:
        print(f"Current sample ID: {current_id}")
    if current_id == 1000:
        break


with open(out_file, 'a') as f:
    f.write(']')
