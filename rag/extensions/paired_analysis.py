"""Paired analysis of the dev answer-quality runs (no ELM calls; standard library only).

Reads the result files in RESULTS_DIR (normally rag/results) for the three dev
configurations (baseline, MiniLM at N=20, bge at N=10) and prints:
  - per configuration: Recall@5 / MRR@10 / NDCG@10 from the retrieval sweep, the
    mean judge scores, the latency of each stage (median, p95, mean) and the prompt size;
  - for each reranker against the baseline, paired over the same 125 questions: the
    mean change in the normalised judge score with a 95% bootstrap confidence interval
    (10,000 resamples, seed 0) and a sign-flip test p-value; that change split by
    whether Recall@5 went up, down or stayed the same; how often the judge score
    changed when the top-5 set did not; and how many rank-1 hits were kept.

Usage (from the repository root):
    python rag/extensions/paired_analysis.py rag/results
"""
import json,statistics as st,random,sys
R=sys.argv[1]
def jl(p): return [json.loads(l) for l in open(p)]
cfg={"dev_baseline":"baseline","dev_rerank_minilm_N20":"minilm-l6@N20","dev_rerank_bge_N10":"bge-base@N10"}
sweep={r["query_id"]:r for r in jl(f"{R}/rerank_sweep_dev/per_query.jsonl")}
J={n:{r["query_id"]:r for r in jl(f"{R}/{n}_judge/answer_judge_per_query.jsonl")} for n in cfg}
G={n:{r["query_id"]:r for r in jl(f"{R}/{n}.jsonl")} for n in cfg}
qs=list(J["dev_baseline"])
assert all(set(J[n])==set(qs)==set(G[n]) for n in cfg)
print("transcripts identical:", all(G[n][q]["transcript"]==G["dev_baseline"][q]["transcript"] for n in cfg for q in qs))
def pct(v,p): v=sorted(v); return v[min(len(v)-1,max(0,int(round(p*len(v)))-1))]
for n,c in cfg.items():
    key="baseline" if c=="baseline" else c
    s=[J[n][q]["normalised_answer_score"] for q in qs]
    print(f"\n== {n}")
    print(f"  R@5 {st.mean(sweep[q][key]['Recall@5'] for q in qs):.4f} MRR {st.mean(sweep[q][key]['MRR@10'] for q in qs):.4f} NDCG {st.mean(sweep[q][key]['NDCG@10'] for q in qs):.4f}")
    print(f"  judge {st.mean(s):.4f} zero {sum(x==0 for x in s)} corr {st.mean(J[n][q]['correctness'] for q in qs):.3f} grd {st.mean(J[n][q]['grounding'] for q in qs):.3f} cmp {st.mean(J[n][q]['completeness'] for q in qs):.3f}")
    for k in ["embedding_ms","retrieval_ms","rerank_ms","generation_ms","end_to_end_ms"]:
        v=[G[n][q]["timing"][k] for q in qs if k in G[n][q]["timing"]]
        if v: print(f"  {k:14} median {st.median(v):8.1f} p95 {pct(v,.95):8.1f} mean {st.mean(v):8.1f}")
    pc=[G[n][q]["prompt_chars"] for q in qs]; print(f"  prompt_chars mean {st.mean(pc):.0f} median {st.median(pc):.0f}")
random.seed(0)
def boot(d,B=10000):
    m=sorted(st.mean(random.choices(d,k=len(d))) for _ in range(B)); return m[int(.025*B)],m[int(.975*B)]
def flip(d,B=10000):
    o=abs(st.mean(d)); return sum(abs(st.mean([x*random.choice((1,-1)) for x in d]))>=o for _ in range(B))/B
b=J["dev_baseline"]
for n,c in list(cfg.items())[1:]:
    d=[J[n][q]["normalised_answer_score"]-b[q]["normalised_answer_score"] for q in qs]
    lo,hi=boot(d); p=flip(d)
    print(f"\n## {c}: dJ {st.mean(d):+.4f} CI [{lo:+.4f},{hi:+.4f}] p={p:.3f} better {sum(x>0 for x in d)} worse {sum(x<0 for x in d)} same {sum(x==0 for x in d)} halfwidth {(hi-lo)/2:.3f}")
    dr=[sweep[q][c]["Recall@5"]-sweep[q]["baseline"]["Recall@5"] for q in qs]
    for lab,f in [("up",lambda x:x>0),("down",lambda x:x<0),("same",lambda x:x==0)]:
        idx=[i for i,x in enumerate(dr) if f(x)]
        print(f"   R@5 {lab:4} n={len(idx):3} dJ {st.mean(d[i] for i in idx):+.3f} better {sum(d[i]>0 for i in idx)} worse {sum(d[i]<0 for i in idx)}")
    sameset=[q for q in qs if set(G[n][q]["retrieved_record_ids"])==set(G["dev_baseline"][q]["retrieved_record_ids"])]
    print("   same top-5 set:",len(sameset),"judge changed:",sum(J[n][q]["normalised_answer_score"]!=b[q]["normalised_answer_score"] for q in sameset))
    base1=[q for q in qs if sweep[q]["baseline"]["MRR@10"]==1.0]
    print("   rank-1: baseline",len(base1),"kept",sum(sweep[q][c]["MRR@10"]==1.0 for q in base1),"total",sum(sweep[q][c]["MRR@10"]==1.0 for q in qs))
