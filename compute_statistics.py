import json
import os
import sys
import numpy as np

path = sys.argv[1]

bleu = []
rouge1 = []
rouge2 = []
rougeL = []

with open(path) as source:
    for line in source:
        entry = json.loads(line)
        bleu.append(entry["metrics"]["bleu"])
        rouge1.append(entry["metrics"]["rouge1f1"])
        rouge2.append(entry["metrics"]["rouge2f1"])
        rougeL.append(entry["metrics"]["rougeLf1"])

    bleu = np.array(bleu)
    rouge1 = np.array(rouge1)
    rouge2 = np.array(rouge2)
    rougeL = np.array(rougeL)

    print(f"Bleu mean: {np.mean(bleu)} std: {np.std(bleu)}")
    print(f"rouge1 mean: {np.mean(rouge1)} std: {np.std(rouge1)}")
    print(f"rouge2 mean: {np.mean(rouge2)} std: {np.std(rouge2)}")
    print(f"rougeL mean: {np.mean(rougeL)} std: {np.std(rougeL)}")