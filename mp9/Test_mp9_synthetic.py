"""End-to-end test of mp9_narms on synthetic NARMS-shaped data, with stub faiss/deal, a hashing
embedder and a fake LLM backend. The simulated 'human' labels below exist ONLY in this test."""
import sys, os, json, tempfile
sys.path[:0] = [os.path.join(os.path.dirname(__file__), "stubs"), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
import numpy as np, pandas as pd
import mp7_core as M, mp7_narms as N, mp9_narms as P

os.chdir(tempfile.mkdtemp())
raw = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "synth_narms.csv"), low_memory=False)
cfg = N.NarmsConfig()
df, drugs, rep = N.prepare(raw, cfg)
print("drugs", drugs, rep["dropped_for_missingness"])
train_df, test_df = N.temporal_split(df, cfg)

class HashEmb:
    model_id, revision = "hash", "0"
    def __init__(s): s.c = {}
    def encode(s, texts):
        out = []
        for t in texts:
            if t not in s.c:
                v = np.zeros(64, "float32")
                for w in t.lower().replace(",", " ").replace(".", " ").split():
                    v[sum(map(ord, w)) % 64] += 1
                s.c[t] = v / max(np.linalg.norm(v), 1e-9)
            out.append(s.c[t])
        return np.stack(out)

class FakeLLM(M.LLMBackend):
    model_id, revision = "fake", "0"
    def check_verbalizers(s, v): pass
    def truncate(s, t, n): return t
    def render(s, sys_, user): return sys_ + "\n" + user + "\nA:"
    def label_logprobs(s, prefix, rest, verb):
        r = prefix.count("Tetracycline: Resistant") + 1; q = prefix.count("Tetracycline: Susceptible") + 1
        sig = 1.5 if "sulfisoxazole" in rest else -1.0
        z = np.log(r / q) + sig
        return np.log(np.array([1 / (1 + np.exp(-z)), 1 / (1 + np.exp(z))]))

emb, llm = HashEmb(), FakeLLM()

# ---- 1. gold
ids = P.select_gold_candidates(test_df, cfg, n=50)
assert len(ids) == 50 and len(set(ids)) == 50
sheet = P.make_annotation_sheet(df, ids, cfg, drugs)
assert (sheet.label == "").all() and "TET SIR" not in sheet.columns
N.assert_no_target_leak([M.Example("x", t, None) for t in sheet.profile])
print("borderline share:", P.is_borderline(sheet.tet_mic_raw).mean())
lab = df.set_index(df[cfg.id_col].astype(str))["label"]
mine = sheet.copy()                                   # SIMULATED annotator (test only)
mine["label"] = mine.item_id.map(lab); mine.loc[3, "label"] = "exclude"
mine["borderline"] = np.where(P.is_borderline(mine.tet_mic_raw), "yes", "no")
mine.to_csv("gold_raw.csv", index=False)
print(P.precheck_gold_raw("gold_raw.csv"))
pd.concat([mine, mine.iloc[:1]]).to_csv("bad.csv", index=False)
try: P.precheck_gold_raw("bad.csv"); raise SystemExit("dup not caught")
except ValueError as e: print("caught:", str(e).splitlines()[1])
big = pd.concat([mine, mine.assign(item_id=mine.item_id + "x")]); big.to_csv("big.csv", index=False)
try: P.precheck_gold_raw("big.csv"); raise SystemExit("cap not caught")
except ValueError as e: print("caught:", str(e).splitlines()[1])
print(P.audit_vs_lab(mine, df, cfg).agree.mean())

# ---- 2. IAA
ov = ids[:20]
partner = mine[mine.item_id.isin(ov)].copy()
flip = partner.index[[0, 5, 9]]
partner.loc[flip, "label"] = partner.loc[flip, "label"].map({"resistant": "susceptible", "susceptible": "resistant", "exclude": "resistant"})
kr = P.kappa_report(mine.set_index("item_id").label.to_dict(), partner.set_index("item_id").label.to_dict(), ov, n_boot=2000)
print({k: v for k, v in kr.items() if k != "confusion_matrix"}); print(kr["confusion_matrix"])
from sklearn.metrics import cohen_kappa_score
a_ = [mine.set_index("item_id").label[i] for i in ov]; b_ = [partner.set_index("item_id").label[i] for i in ov]
assert abs(kr["kappa"] - cohen_kappa_score(a_, b_)) < 1e-12
try: P.adjudicate(mine, partner, ov, {}); raise SystemExit("missing decision not caught")
except ValueError as e: print("caught:", e)
dec = {i: {"label": mine.set_index("item_id").label[i] if mine.set_index("item_id").label[i] != "exclude" else "resistant",
           "rule": "R2", "note": "test"} for i in kr["disagreements"]}
adj = P.adjudicate(mine, partner, ov, dec)
gold = P.gold_examples(df, cfg, drugs, "gold_adjudicated.jsonl")
print(len(gold), adj.adjudication.value_counts().to_dict())

# ---- 3. weak supervision
pool_all = N.to_examples(train_df, cfg, drugs)
tr = train_df.sort_values(cfg.id_col)
rng = np.random.default_rng(P.SEED)
is_dev = rng.random(len(tr)) < 0.2
dev_df, unl_df = tr[is_dev], tr[~is_dev]
silver_df = unl_df.iloc[np.sort(rng.choice(len(unl_df), 1500, replace=False))]
fdev, fsil = P.lf_frame(dev_df, cfg, drugs), P.lf_frame(silver_df, cfg, drugs)
fgold = P.lf_frame(df[df[cfg.id_col].astype(str).isin([g.item_id for g in gold])], cfg, drugs)
ydev = dev_df.set_index(dev_df[cfg.id_col].astype(str)).label
dev_ex = N.to_examples(dev_df, cfg, drugs)
knn = M.KNNClassifier(emb, n_neighbors=5).fit(M.sample_demos(dev_ex, 8, 0))
texts_all = {e.item_id: e.text for e in N.to_examples(pd.concat([dev_df, silver_df]), cfg, drugs) + gold}
ids_all = list(texts_all)
mp7_preds = dict(zip(ids_all, knn.predict([texts_all[i] for i in ids_all])))
lfs = P.make_default_lfs(fdev, ydev, mp7_preds)
Ldev, Lsil, Lgold = P.apply_lfs(fdev, lfs), P.apply_lfs(fsil, lfs), P.apply_lfs(fgold, lfs)
print(P.lf_summary(Ldev, ydev))
w = P.fit_lf_weights(Ldev, ydev); print("weights", {k: round(v, 3) for k, v in w.items()})
vs, vg = P.weighted_vote(Lsil, w), P.weighted_vote(Lgold, w)
print("silver written:", P.silver_records(vs, Lsil))
assert all(json.loads(l)["source"] == "silver" for l in open("silver_pool.jsonl"))
gl = pd.Series({g.item_id: g.label for g in gold})
ag = P.agreement(gl, vg.silver_label.reindex(gl.index)); print({k: v for k, v in ag.items() if k != "confusion_matrix"})
ysil = silver_df.set_index(silver_df[cfg.id_col].astype(str)).label
print("silver vs lab on pool:", {k: v for k, v in P.agreement(ysil, vs.silver_label).items() if k != "confusion_matrix"})

# ---- 4. retrieval-augmented classification
idx = P.ExemplarIndex(emb, pool_all); idxb = P.ExemplarIndex(emb, pool_all, by_label=True)
h = idx.search(gold[0].text, 5); assert all(x[1] >= y[1] for x, y in zip(h, h[1:]))
assert not ({e.item_id for e in pool_all} & {g.item_id for g in gold})
c = P.RetrievalFewShotLLMClassifier(llm, idxb, 3, balanced=True).fit([])
c.predict_proba([gold[0].text]); labs = [ {e.item_id: e.label for e in pool_all}[i] for i, _ in c.retrieved[0]]
print("balanced k=3 labels:", labs); assert len(set(labs)) == 2
for k in (3, 5, 10):
    d = P.sample_demos_total(pool_all, k, 0); assert len(d) == k
    assert [x.item_id for x in d] == [x.item_id for x in P.sample_demos_total(pool_all, k, 0)]
runs = P.run_rac_sweep(llm, idx, idxb, pool_all, gold, ks=(3, 5, 10), seeds=(0, 1))
tab = P.metric_table("predictions_mp9.jsonl", n_boot=300); print(tab.round(3).to_string())
print(P.paired_bootstrap_diff("predictions_mp9.jsonl", "retrieval_llm", "fixed_llm", 5, n_boot=500))
t2, best, per = M.summarize("predictions_mp9.jsonl", metric=M.dev_macro_f1, n_boot=200); print(best.classifier, best.k)

ranked = P.ranked_neighbors(idx, gold, 10)
fpool = P.lf_frame(train_df, cfg, drugs)
pool_lab = {e.item_id: e.label for e in pool_all}; gold_lab = {g.item_id: g.label for g in gold}
is_rel, n_tot = P.rule_relevance(fgold, fpool, gold_lab, pool_lab)
rel = {q: {c: int(is_rel(q, c)) for c in cs} for q, cs in ranked.items()}
tot = {q: n_tot(q) for q in ranked}
assert all(tot[q] >= sum(rel[q].values()) for q in ranked)
print(P.retrieval_metrics(ranked, rel, n_relevant_total=tot, n_boot=500).round(3).to_string())
rs = P.relevance_sheet(ranked, fpool, {e.item_id: e.text for e in pool_all}, {g.item_id: g.text for g in gold})
rs.assign(relevant=[rel[q][c] for q, c in zip(rs.query_id, rs.cand_id)]).to_csv("rel.csv", index=False)
assert P.read_relevance("rel.csv") == rel

# ---- 5. convolution
F = np.array([[0., 10], [1, 12], [3, 11], [6, 15]])
assert np.allclose(P.apply_kernel(F, P.DIFF), np.diff(F, axis=0))
assert np.allclose(P.apply_kernel(F, P.SMOOTH)[0], .25 * F[0] + .5 * F[1] + .25 * F[2])
dcols = ["TET SIR"] + drugs
dr = P.drift_analysis(df, cfg, dcols, n_boot=200)
print(dr["magnitude"].round(2).to_string()); print(dr.get("policy_excess_drift_pp"))
print(dr["per_drug"].query("drug == 'TET SIR'").round(2).to_string())

# ---- 6. learned kernel (lstsq path; torch is not installed in this sandbox)
fc = P.forecast_comparison(dr["F"], dr["years"], w=3, n_test_years=2, use_torch=False, n_boot=500)
print(fc["table"].round(3).to_string()); print(fc["kernels"])

P.write_results("results.json", {"iaa": {k: v for k, v in kr.items()}, "silver_vs_gold": ag, "rac": tab,
                                 "drift": {k: dr[k] for k in ("magnitude", "per_drug", "years", "drugs")},
                                 "forecast": fc["table"]})
json.load(open("results.json")); print("results.json OK")
print("ALL TESTS PASSED")