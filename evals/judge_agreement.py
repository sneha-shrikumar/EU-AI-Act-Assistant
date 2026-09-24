"""Judge-vs-human agreement, TPR and TNR on the hand-labelled first run (afc59a56).

Replays the relevance and correctness judges over every answer in
results/baseline-handscored-afc59a56.csv, where each answer was scored by hand,
and compares. "Positive" = pass (score 1); TPR = share of human passes the judge
also passes; TNR = share of human fails the judge also fails. Groundedness is
not replayed: that run predates storing the retrieved excerpts.

    python evals/judge_agreement.py evals/results/judge_vs_human_afc59a56.json

Costs about 82 judge calls.
"""
import csv, json, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "."); sys.path.insert(0, "evals")
from dotenv import load_dotenv; load_dotenv(".env")
from evals.judges import answer_relevance, answer_correctness
rows = list(csv.DictReader(open("results/baseline-handscored-afc59a56.csv", encoding="utf-8-sig")))
def run(x):
    i = json.loads(x["inputs"]); o = json.loads(x["outputs"]); ref = json.loads(x["reference_outputs"])
    out = {"q": i["question"], "human": {"answer_relevance": int(x["Answer Relevance"]), "answer_correctness": int(x["Answer correctness"])}}
    for f in (answer_relevance, answer_correctness):
        r = f(i, o, ref); out[f.__name__] = r.get("score"); out[f.__name__ + "_c"] = r.get("comment")
    return out
with ThreadPoolExecutor(2) as ex: res = list(ex.map(run, rows))
json.dump(res, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
for k in ("answer_relevance", "answer_correctness"):
    tp=fn=tn=fp=none=0
    for r in res:
        h, j = r["human"][k], r[k]
        if j is None: none += 1; continue
        if h==1 and j==1: tp+=1
        elif h==1 and j==0: fn+=1
        elif h==0 and j==0: tn+=1
        else: fp+=1
    n=tp+fn+tn+fp
    print(k, dict(tp=tp,fn=fn,tn=tn,fp=fp,none=none), "agree", round((tp+tn)/n,3), "TPR", round(tp/(tp+fn),3), "TNR", round(tn/(tn+fp),3))
    for r in res:
        if r[k] is not None and r[k]!=r["human"][k]: print("   disagree: human",r["human"][k],"judge",r[k],"|",r["q"][:70])
