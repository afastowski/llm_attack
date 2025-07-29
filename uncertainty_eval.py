from datasets import load_dataset
import numpy as np
from collections import defaultdict
from collections import Counter
import argparse
import math

parser = argparse.ArgumentParser(description="Running uncertainty metrics.")
parser.add_argument("-m", "--model", type=str, default="phi", help="LLM to query.")
parser.add_argument("-d", "--dataset", type=str, default="nq", help="Dataset to evaluate. triviaqa, squad, or nq.")
parser.add_argument("-v", "--version", type=int, default=3, help="Prompt v0, v1, v2 or v3.")
# v0 ="baseline". I.e. no manipulation.
# v1 = "Respond with a **wrong**, exact answer only."
# v2 = the false_info x k setting.
# v3 = the "random info" setting.

args = parser.parse_args()

if args.version == 0:
    mode = "baseline"
elif args.version == 1:
    mode = "direct"
elif args.version == 2:
    mode = "false"
elif args.version == 3:
    mode = "random"


if mode == "baseline":
    read_file = f"scores_1000/{args.dataset}/{args.model}/uncertainties_baseline.json"
else:
    read_file = f"scores_1000/{args.dataset}/{args.model}/v{args.version}/uncertainties_{mode}_prompt.json"

ds_uncertainties = load_dataset("json", data_files=read_file)
uncertainties = defaultdict(list)


for sample in ds_uncertainties["train"]:
    ae = np.mean(sample["uncertainty"]["ae"])
    ppl = np.mean(sample["uncertainty"]["ppl"])
    ap = np.mean(sample["uncertainty"]["ap"])

    counter = Counter(sample["model_answer"])
    model_answer, frequency = counter.most_common(1)[0]
    uncertainties["model_answer"].append(model_answer)
    uncertainties["correct_answer"].append(sample["answer"])

    uncertainties["ae"].append(ae)
    uncertainties["ppl"].append(ppl)
    uncertainties["ap"].append(ap)

# print("Overall Uncertainty.")
# print()
num_results = len(uncertainties["ae"])
ae = np.mean(uncertainties["ae"])
ae_std_err =np.std(uncertainties["ae"]) / np.sqrt(num_results)
ppl = np.mean(uncertainties["ppl"])
ppl_std_err = np.std(uncertainties["ppl"]) / np.sqrt(num_results)
ap = np.mean(uncertainties["ap"])
ap_std_err = np.std(uncertainties["ap"]) / np.sqrt(num_results)

# print("Entropy:", f"${round(ae,2)}$")
# print("PPL:", f"${round(ppl,2)}$")
# print("TP:", f"${round(ap,2)}$")

## Measure Answer Accuracy.

correct = 0

correct_answers = defaultdict(list)
incorrect_answers = defaultdict(list)

for i, example in enumerate(uncertainties["ae"]):
    if uncertainties["correct_answer"][i].lower() in uncertainties["model_answer"][i].lower():
        correct += 1
        is_correct = uncertainties["correct_answer"][i].lower() in uncertainties["model_answer"][i].lower()

        
        correct_answers["ae"].append(uncertainties["ae"][i])
        correct_answers["ppl"].append(uncertainties["ppl"][i])
        correct_answers["ap"].append(uncertainties["ap"][i])

    # if false prompt changed the answer to a wrong one
    else:
        #print(uncertainties["correct_answer"][i], "---", uncertainties["model_answer"][i])
        incorrect_answers["ae"].append(uncertainties["ae"][i])
        incorrect_answers["ppl"].append(uncertainties["ppl"][i])
        incorrect_answers["ap"].append(uncertainties["ap"][i])
acc = correct/num_results
print("Overall Accuracy: ", round(correct/num_results, 4))
print("Std Err:",  round(math.sqrt((acc * (1 - acc)) / num_results),2))
print()

# ##################
# #print("Correct Answers Metrics:")

# correct_num_results = len(correct_answers["ae"])

# correct_ae = np.mean(correct_answers["ae"])
# correct_ae_std_err =np.std(correct_answers["ae"]) / np.sqrt(correct_num_results)

# correct_ppl = np.mean(correct_answers["ppl"])
# correct_ppl_std_err = np.std(correct_answers["ppl"]) / np.sqrt(correct_num_results)

# correct_ap = np.mean(correct_answers["ap"])
# correct_ap_std_err = np.std(correct_answers["ap"]) / np.sqrt(correct_num_results)

# # print("Correct Ratio: ", correct_num_results / num_results)
# # print("Entropy: ", f"${round(correct_ae,2)}^{{\pm{round(correct_ae_std_err,3)}}}$")
# # print("PPL: ", f"${round(correct_ppl,2)}^{{\pm{round(correct_ppl_std_err,3)}}}$")
# # print("Prob: ", f"${round(correct_ap,2)}^{{\pm{round(correct_ap_std_err,3)}}}$")
# # print()

# ###################
# #print("Incorrect Answers Metrics:")

# incorrect_num_results = len(incorrect_answers["ae"])

# incorrect_ae = np.mean(incorrect_answers["ae"])
# incorrect_ae_std_err =np.std(incorrect_answers["ae"]) / np.sqrt(incorrect_num_results)

# incorrect_ppl = np.mean(incorrect_answers["ppl"])
# incorrect_ppl_std_err = np.std(incorrect_answers["ppl"]) / np.sqrt(incorrect_num_results)

# incorrect_ap = np.mean(incorrect_answers["ap"])
# incorrect_ap_std_err = np.std(incorrect_answers["ap"]) / np.sqrt(incorrect_num_results)

# # print("Incorrect Ratio: ", incorrect_num_results / num_results)
# # print("Entropy: ", f"${round(incorrect_ae,2)}^{{\pm{round(incorrect_ae_std_err,3)}}}$")
# # print("PPL: ", f"${round(incorrect_ppl,2)}^{{\pm{round(incorrect_ppl_std_err,3)}}}$")
# # print("Prob: ", f"${round(incorrect_ap,2)}^{{\pm{round(incorrect_ap_std_err,3)}}}$")

# # print(f"&  & ${round(correct_ae,2)}^{{\pm{round(correct_ae_std_err,3)}}}$ & ${round(correct_ppl,2)}^{{\pm{round(correct_ppl_std_err,3)}}}$ & ${round(correct_ap,2)}^{{\pm{round(correct_ap_std_err,3)}}}$ &  & ${round(incorrect_ae,2)}^{{\pm{round(incorrect_ae_std_err,3)}}}$ & ${round(incorrect_ppl,2)}^{{\pm{round(incorrect_ppl_std_err,3)}}}$ & ${round(incorrect_ap,2)}^{{\pm{round(incorrect_ap_std_err,3)}}}$")
