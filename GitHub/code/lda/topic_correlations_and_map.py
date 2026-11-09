"""Doc-topic correlations and pyLDAvis data for the reference K = 3 model (seed 1234, 2,000 iterations)."""
import json
import numpy as np
import pandas as pd
import pyLDAvis
import lda_bibliometric_analysis_v2 as base
from lda_sensitivity import build_corpus
M, Mp, dtm, feat = build_corpus()
lda, dt = base.run_lda_simple(dtm, 3)   # same call as the main script (n_jobs=-1), to reproduce the saved model
ref = pd.read_excel(base.config.results_dir / "03_lda_results.xlsx", sheet_name="Documents with Topics")
print("same dominant topic as saved results:", float((np.argmax(dt, axis=1) + 1 == ref["main_topic"].values).mean()), np.bincount(np.argmax(dt, axis=1)))
corr = np.corrcoef(dt.T)
print(pd.DataFrame(corr, index=range(1, 4), columns=range(1, 4)).round(3).to_string())
ttd = lda.components_ / lda.components_.sum(axis=1)[:, None]
vis = pyLDAvis.prepare(topic_term_dists=ttd, doc_topic_dists=dt, doc_lengths=np.array(dtm.sum(axis=1)).flatten(),
                       vocab=list(feat), term_frequency=np.array(dtm.sum(axis=0)).flatten(), sort_topics=False)
tc = vis.topic_coordinates.reset_index()
info = vis.topic_info[vis.topic_info.Category == "Default"].sort_values("Freq", ascending=False)
out = dict(mds=dict(topics=[int(t) for t in tc["topics"]], x=tc["x"].tolist(), y=tc["y"].tolist(), Freq=tc["Freq"].tolist()),
           salient=[[t, float(f)] for t, f in zip(info.Term, info.Freq)][:30], corr=corr.tolist())
json.dump(out, open(base.config.results_dir / "ldavis_figure5_data.json", "w"))
print(out["mds"]); print(out["salient"][:30])
