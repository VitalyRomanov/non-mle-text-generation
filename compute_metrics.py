from bleurt import score as bleurt_score
from bert_score import score as bert_score_score
from statistics import mean
import argparse
import sacrebleu
import numpy as np

references_dummy = ["Bud Powell was a legendary pianist.",
                  "Bud Powell was a legendary pianist.",
                  "Bud Powell was a legendary pianist.",
                  "Bud Powell was a legendary pianist."]
candidates_dummy = ["Bud Powell was a legendary pianist.",
                  "Bud Powell was a historical piano player.",
                  "Bud Powell was a new yorker.",
                  "Bud a day keeps the doctor away."]
true_sents_dummy = ["Piano music is good",
                    "There is a historical piano at the city center museum.",
                    "Bud Powell is some well-known man.",
                    "An apple a day keeps the doctor away."
                    "Some irrelevant text.",
                    "Some more irrelevant text.",
                    "It was a legendary move!"
                    ]


# BLEURT, automatic relevance metric
def bleurt_eval(candidates, references, verbose=False):
    checkpoint = "bleurt/bleurt-base-128"
    scorer = bleurt_score.BleurtScorer(checkpoint)
    scores = scorer.score(references=references, candidates=candidates)
    if verbose:
        print("BLEURT scores:", scores)
    return mean(scores)


# BertScore, another automatic relevance metric
def bert_score_eval(candidates, references, verbose=False):
    P, R, F1 = bert_score_score(candidates, references, lang='en', verbose=verbose, device='cpu')
    if verbose:
        print("BertScores:")
        print("P:", P)
        print("R:", R)
        print("F1:", F1)
    P_avg, R_avg, F1_avg = P.mean().numpy(), R.mean().numpy(), F1.mean().numpy()
    return P_avg, R_avg, F1_avg


# BLEU, from sacrebleu
def usual_bleu_eval(candidates, references):
    usual_bleu = sacrebleu.corpus_bleu(candidates, [references])
    return usual_bleu


# CORPUS_BLEU, how similar are translations to some ground true text
# kinda fluency
def corpus_bleu_eval(candidates, ground_true_sents):
    # all true sents for each candidate, NB: won't work properly if there are same sents in candidates and ground_true
    true_sents_for_each_candidate = [[sent for c in candidates] for sent in ground_true_sents]
    corpus_bleu = sacrebleu.corpus_bleu(candidates, true_sents_for_each_candidate)
    return corpus_bleu


# SELF_BLEU, measures diversity
def self_bleu_eval(candidates):
    # all OTHER candidates for each candidate, NB: won't work properly if there are duplicates in candidates
    candidates_for_each_candidate = []
    n_cands = len(candidates)
    for i in range(n_cands-1):
        ith_and_last = [candidates[i] for j in range(n_cands)]
        ith_and_last[i] = candidates[-1]
        candidates_for_each_candidate.append(ith_and_last)
    self_bleu = sacrebleu.corpus_bleu(candidates, candidates_for_each_candidate)
    return self_bleu

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cand_file", default=None)  # candidate translations (generated)
    parser.add_argument("--ref_file", default=None)  # reference translations (ground truth)
    parser.add_argument("--true_corpus_file", default=None)  # real text, for corpus-bleu.
    # if real text has 2k sents, it takes 0.3 sec/candidate, if 6k sents, 0.6 sec/candidate
    parser.add_argument("--verbose", default=0)  # prints scores for each sent (bleurt & bertscore) + some logs
    args = parser.parse_args()
    if args.ref_file is None:
        print("Using dummy files for references and candidates")
        references = references_dummy
        candidates = candidates_dummy
    else:
        with open(args.ref_file) as f:
            references = [line.strip() for line in f]
        with open(args.cand_file) as f:
            candidates = [line.strip() for line in f]

    if args.true_corpus_file is None:
        print("Using dummy file for true_sents")
        true_sents = true_sents_dummy
    else:
        with open(args.ref_file) as f:
            true_sents = [line.strip() for line in f]

    BOLD = '\033[1m'
    END = '\033[0m'
    bleurt = bleurt_eval(candidates, references, args.verbose == 1)
    print(BOLD + "BLEURT score:" + END, np.round(bleurt, 4))
    P_avg, R_avg, F1_avg = bert_score_eval(candidates, references, args.verbose == 1)
    print(BOLD + "BERTSCORE: P_avg, R_avg, F1_avg:" + END, np.round(P_avg, 4), np.round(R_avg, 4), np.round(F1_avg, 4))
    usual_bleu = usual_bleu_eval(candidates, references)
    print(BOLD + "Usual BLEU:" + END, usual_bleu)
    corpus_bleu = corpus_bleu_eval(candidates, true_sents)
    print(BOLD + "Corpus BLEU (fluency):" + END, corpus_bleu)
    self_bleu = self_bleu_eval(candidates)
    print(BOLD + "Self-BLEU (diversity):" + END, self_bleu)