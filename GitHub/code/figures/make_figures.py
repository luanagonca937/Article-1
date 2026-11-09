"""Redraws every figure of the article with one visual system (journal print, light surface).

v13: Figure 2 coloured by LDA topic; Figure 7 as an author x topic bubble matrix;
Figure 8 as a three-field Sankey (author keywords -> topics -> sources, matplotlib).
v16 (reviewer): Figure 1 says "expert review" instead of "validation" and "exploratory literature mapping"
instead of "systematic review"; new record selection flow diagram (fig_flow_selection.png).
v17 (reviewer): composite index shown as a descriptive ranking (thirds, not priorities); K-means clusters
removed from the alignment matrix; "mentions identified by the dictionary" instead of "evidenced".
v18: scope of the cases: agro-industrial cooperatives with poultry operations.
v19: collection of 1 November 2026, K = 5.
v33: IEEE record with abstract, K = 3; focal topic is Topic 1.
v40: 600 dpi, so that every figure exceeds 300 dpi at the size used in the manuscript (journal requirement)."""
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

RES = Path(sys.argv[1])          # results_v2_corrigido
OUT = Path(sys.argv[2]); OUT.mkdir(exist_ok=True)
HERE = Path(__file__).parent

# ---- design tokens ----
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
GRID, SURF, EMPTY = "#e4e3df", "#ffffff", "#f0efec"
TOPIC = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
TNAME = {1: "Digital transformation & management", 2: "Policy, governance & environment",
         3: "Systems, energy & health models"}
TOPICS = range(1, 4)
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
FOCAL = TOPIC[1]

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})
W = 6.5  # inches, full text width

def grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6); ax.set_axisbelow(True)

# Figure language: the public repository generates the English manuscript figures.
# The internal controlled archive retains the optional Portuguese translation resources.
LANG = sys.argv[3] if len(sys.argv) > 3 else "en"
LABELS = {}

def to_brazilian_numbers(s):
    """1,688 -> 1.688 and 7.518 -> 7,518. A number right after a word ('Industry 4.0', 'Lean 4.0' in article
    titles) is part of a name, not a value, and is kept as written."""
    def convert(m):
        return m.group(0).replace(",", "\0").replace(".", ",").replace("\0", ".")
    return re.sub(r"(?<![A-Za-z] )(?<![\w.,])(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+)", convert, s)

def localize(fig, name):
    """Before saving: translate every text of the figure and use decimal commas on numeric axes (pt)."""
    import matplotlib.text as mtext
    from matplotlib.ticker import ScalarFormatter
    if LANG == "dump":
        LABELS[name] = sorted({t.get_text() for t in fig.findobj(mtext.Text) if t.get_text().strip()})
        return
    from matplotlib.ticker import FixedFormatter
    tr = json.loads((HERE / "figure_labels_pt.json").read_text(encoding="utf-8"))
    partial = sorted(tr["partial"].items(), key=lambda kv: len(kv[0]), reverse=True)

    def translate(s):
        if not s.strip():
            return s
        s = to_brazilian_numbers(s)
        if s in tr["exact"]:
            return tr["exact"][s]
        for en, pt in partial:
            s = s.replace(en, pt)
        return s

    for t in fig.findobj(mtext.Text):
        t.set_text(translate(t.get_text()))
    from matplotlib.ticker import FuncFormatter
    for ax in fig.axes:   # text tick labels are redrawn from the formatter at save time: wrap it
        for axis in (ax.xaxis, ax.yaxis):
            fmt = axis.get_major_formatter()
            if isinstance(fmt, FixedFormatter):
                fmt.seq = [translate(str(x)) for x in fmt.seq]
            elif isinstance(fmt, FuncFormatter):
                axis.set_major_formatter(FuncFormatter(lambda v, p, f=fmt: translate(f(v, p))))
    class CommaFormatter(ScalarFormatter):
        """Same tick labels as matplotlib's default (same number of decimals), with a decimal comma."""
        def __call__(self, x, pos=None):
            return super().__call__(x, pos).replace(".", ",")
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if type(axis.get_major_formatter()) is ScalarFormatter:
                axis.set_major_formatter(CommaFormatter())

def save(fig, name):
    if LANG != "en":
        localize(fig, name)
    fig.savefig(OUT / name); plt.close(fig); print("saved", name)
    if LANG == "dump":
        (OUT / "figure_labels.json").write_text(json.dumps(LABELS, ensure_ascii=False, indent=1), encoding="utf-8")

def clean(s, n=None):
    s = str(s).replace("\ufffd", "–").replace("  ", " ").strip()
    return (s[: n - 1] + "…") if n and len(s) > n else s

def hbar(ax, labels, values, color, fmt="{:.0f}", pad=0.02):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, height=0.62, color=color, edgecolor=SURF, linewidth=1)
    vmax = max(values) if len(values) else 1
    for yi, v in zip(y, values):
        ax.text(v + vmax * pad, yi, fmt.format(v), va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    grid(ax, "x")

# ---------- Figure 1: research design ----------
def fig1():
    fig, ax = plt.subplots(figsize=(W, 5.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    def box(x, y, w, h, text, fc, ec, fs=7.5, bold=False, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=0.9, ls=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=INK, linespacing=1.25)
    def arrow(x0, y0, x1, y1, color=MUTED, ls="-"):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9, ls=ls,
                                                             shrinkA=0, shrinkB=0, mutation_scale=8))
    def band(y, h, title, sub, ec, fc, ls="-"):
        ax.add_patch(FancyBboxPatch((1, y), 98, h, boxstyle="round,pad=0.3,rounding_size=1.5", fc=fc, ec=ec, lw=1, ls=ls))
        ax.text(3, y + h - 3.2, title, fontsize=8.5, fontweight="bold", color=INK, va="center")
        ax.text(97, y + h - 3.2, sub, fontsize=7.5, color=INK2, va="center", ha="right")
    # Layer 1
    band(66, 32, "Layer 1 – Exploratory literature mapping with LDA topic modelling", "answers SQ1", "#3987e5", "#f4f8fe")
    steps = ["Database search\nScienceDirect, Scopus,\nWoS, IEEE Xplore\n1,849 records",
             "Document-type\nfilter\n1,287 articles",
             "Title-based\ndeduplication\n1,272 articles",
             "Pre-processing\n(abstract ≥ 100\ncharacters)\n1,249 documents",
             "LDA (K = 3) +\ncomposite topic\nranking"]
    bw, gap, x0 = 16.4, 2.6, 3
    for i, s in enumerate(steps):
        x = x0 + i * (bw + gap)
        box(x, 70, bw, 18, s, SURF, "#86b6ef", fs=6.3)
        if i: arrow(x - gap + 0.2, 79, x - 0.2, 79)
    # Layer 1b: Hoshin alignment
    box(8, 51, 84, 8, "Hoshin Kanri alignment matrix: Topic 1 (502 articles) × 20 guidelines (FAIR framework)",
        "#eef5fd", FOCAL, fs=7.2)
    arrow(50, 66, 50, 59.6, INK2)
    ax.text(51.2, 62.8, "themes and gaps", fontsize=7, color=INK2, va="center")
    # Synthesis
    band(18, 26, "Synthesis and results", "", "#5c5b57", "#f7f7f5")
    outs = ["(i) Implementation\nbarriers and\nresearch gaps", "(ii) Adaptations to the\nFAIR-based Hoshin\nKanri framework",
            "(iii) Integration of\ncultural, organisational\nand technological\ndimensions", "(iv) Exploratory\npropositions for the\ncooperatives"]
    ow, og = 21.5, 2.5
    for i, s in enumerate(outs):
        box(3 + i * (ow + og), 21, ow, 15, s, SURF, "#b9b8b2", fs=6.8)
    arrow(50, 51, 50, 44.6, INK2)
    # feedback loop
    ax.annotate("", (97.5, 72), (97.5, 36), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8, ls=(0, (3, 2)),
                connectionstyle="arc3,rad=-0.25", mutation_scale=8))
    ax.text(99.8, 54, "refinement\nloop", fontsize=6.5, color=INK2, ha="left", va="center")
    # Empirical phase (completed): expert review first, then application; it feeds the synthesis
    band(1, 13, "Layer 2 – Empirical phase", "answers SQ2 and SQ3", "#3987e5", "#f4f8fe")
    box(3, 2.2, 42, 5.2, "Stage 1 – Expert review and pretest:\n5 experts (agreement, clarity, usefulness)",
        SURF, "#86b6ef", fs=6.5)
    box(55, 2.2, 42, 5.2, "Stage 2 – Questionnaire + interviews in 2 agro-\nindustrial cooperatives with poultry operations",
        SURF, "#86b6ef", fs=6.5)
    arrow(46.2, 4.8, 53.8, 4.8)
    arrow(50, 14.6, 50, 18, INK2)
    save(fig, "fig01_research_design.png")

# ---------- Figure 2: word cloud ----------
def docs_with_topics():
    d = pd.read_excel(RES / "03_lda_results.xlsx", sheet_name="Documents with Topics")
    d["text"] = (d["title"].fillna("") + " " + d["abstract"].fillna("")).str.lower()
    return d

def topic_of_word(keyword_set, d):
    """Topic in which each word is relatively most frequent (occurrences per document of the topic)."""
    n = d["main_topic"].value_counts()
    out = {}
    for w in keyword_set:
        term = {"industry 4.0": r"industry\s*4\.?0", "industry 5.0": r"industry\s*5\.?0"}.get(w, re.escape(w))
        cnt = d["text"].str.count(rf"\b{term}\b").groupby(d["main_topic"]).sum()
        rate = {t: cnt.get(t, 0) / n[t] for t in n.index}
        out[w] = max(rate, key=rate.get) if max(rate.values()) > 0 else None
    return out

def fig2():
    from wordcloud import WordCloud
    wf = pd.read_excel(RES / "00_word_frequency_list.xlsx").head(150)
    freqs = dict(zip(wf["word"].astype(str), wf["frequency"].astype(int)))
    freqs = {("industry 4.0" if k == "industry4" else "industry 5.0" if k == "industry5" else k): v for k, v in freqs.items()}
    word_topic = topic_of_word(list(freqs), docs_with_topics())
    def color(word, **kw):
        return TOPIC.get(word_topic.get(word), MUTED)
    # Round cloud, high-resolution circular mask, no contour line; colour = dominant LDA topic.
    size = 2000
    y, x = np.ogrid[:size, :size]
    mask = (((x - size / 2) ** 2 + (y - size / 2) ** 2) > (size / 2 - 20) ** 2).astype(np.uint8) * 255
    wc = WordCloud(width=size, height=size, mask=mask, contour_width=0, background_color=SURF,
                   prefer_horizontal=0.9, max_words=150, font_path=next((f for f in ("C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf") if Path(f).exists()), None),
                   color_func=color, random_state=1234, margin=6,
                   relative_scaling=0.45).generate_from_frequencies(freqs)
    fig, ax = plt.subplots(figsize=(W * 0.75, W * 0.85)); ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
    hand = [plt.Line2D([0], [0], marker="o", ls="", ms=7, color=TOPIC[t]) for t in TOPICS]
    ax.legend(hand, [f"T{t} · {TNAME[t]}" for t in TOPICS], loc="upper center", bbox_to_anchor=(0.5, 0.0),
              ncol=2, frameon=False, fontsize=7, handletextpad=0.3, columnspacing=1.2)
    save(fig, "fig02_wordcloud.png")
    summary = pd.Series(word_topic).value_counts().sort_index()
    print("   words per topic in the cloud:", summary.to_dict())

# ---------- Figure 3: K selection ----------
def fig3():
    k = pd.read_excel(RES / "02_k_metrics.xlsx")
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5))
    for ax, col, lab, fmt in [(axes[0], "Log_Perplexity", "Log perplexity (lower is better)", "{:.3f}"),
                              (axes[1], "Topic_Coherence", "Topic coherence (higher is better)", "{:.3f}")]:
        ax.plot(k["K"], k[col], color=TOPIC[1], lw=2, marker="o", ms=5, mec=SURF, mew=1.2, zorder=3)
        sel = k[k["K"] == 3].iloc[0]
        ax.axvline(3, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.plot([3], [sel[col]], marker="o", ms=8, color=INK, mec=SURF, mew=1.5, zorder=4)
        ax.annotate(f"K = 3: {fmt.format(sel[col])}", (3, sel[col]), xytext=(9, -3 if col == "Log_Perplexity" else 5),
                    textcoords="offset points", fontsize=7.5, color=INK, va="top" if col == "Log_Perplexity" else "bottom")
        lo, hi = k[col].min(), k[col].max(); ax.set_ylim(lo - (hi - lo) * 0.12, hi + (hi - lo) * 0.12)
        ax.set_xlabel("Number of topics (K)"); ax.set_ylabel(lab); ax.set_xticks(k["K"]); grid(ax, "y")
    fig.tight_layout(w_pad=3)
    save(fig, "fig03_k_selection.png")

# ---------- Figure 4: priority ----------
def fig4():
    pr = pd.read_excel(RES / "03_lda_results.xlsx", sheet_name="Topic_Priority").sort_values("Composite_Score", ascending=False)
    fig, ax = plt.subplots(figsize=(W, 2.0))
    labels = [f"Topic {t}  ·  {TNAME[t]}" for t in pr["Topic"]]
    hbar(ax, labels, pr["Composite_Score"].tolist(), [TOPIC[t] for t in pr["Topic"]], fmt="{:.1f}", pad=0.01)
    for thr, txt in [(33, "middle third ≥ 33"), (66, "upper third ≥ 66")]:
        ax.axvline(thr, color=INK2, lw=0.8, ls=(0, (3, 2)))
        ax.text(thr + 0.8, 2.55, txt, fontsize=7, color=INK2, va="bottom")
    ax.set_xlim(0, 100); ax.set_ylim(-0.6, 2.95); ax.set_xlabel("Composite index (descriptive ranking, 0–100)")
    save(fig, "fig04_topic_priority.png")

# ---------- Figure 5: intertopic map + salient terms ----------
def fig5():
    dv = json.load(open(RES / "ldavis_figure5_data.json"))
    mds = dv["mds"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    for t, x, y, f in zip(mds["topics"], mds["x"], mds["y"], mds["Freq"]):
        a1.scatter(x, y, s=f * 52, color=TOPIC[t], alpha=0.9, edgecolor=SURF, linewidth=2, zorder=3)
        a1.text(x, y, f"T{t}", ha="center", va="center", fontsize=9, fontweight="bold", color=SURF, zorder=4)
        dx, dy = 0, 0.105 if y >= 0 else -0.105
        a1.text(x + dx, y + dy, f"{TNAME[t]}\n{f:.1f}% of tokens", ha="center", va="center", fontsize=6.8, color=INK2, zorder=4)
    a1.axhline(0, color=GRID, lw=0.8, zorder=1); a1.axvline(0, color=GRID, lw=0.8, zorder=1)
    a1.set_xlim(min(mds["x"]) - 0.16, max(mds["x"]) + 0.16); a1.set_ylim(min(mds["y"]) - 0.2, max(mds["y"]) + 0.2)
    a1.set_xlabel("PC1"); a1.set_ylabel("PC2"); a1.set_title("Intertopic distance map (MDS)", fontsize=8.5, loc="left", color=INK)
    terms = dv["salient"][:30]
    lab = [("industry 4.0" if t == "industry4" else t) for t, _ in terms]
    vals = [v for _, v in terms]
    y = np.arange(len(lab))[::-1]
    a2.barh(y, vals, height=0.68, color="#86b6ef", edgecolor=SURF, linewidth=0.8)
    a2.set_yticks(y); a2.set_yticklabels(lab, fontsize=6.8); a2.tick_params(axis="y", length=0)
    a2.spines["left"].set_visible(False); grid(a2, "x")
    a2.set_xlabel("Overall term frequency"); a2.set_title("Top-30 most salient terms", fontsize=8.5, loc="left", color=INK)
    fig.tight_layout(w_pad=2)
    save(fig, "fig05_intertopic_map.png")

# ---------- Figure 6: correlation ----------
def fig6():
    # document-topic correlations of the reference model, written by v19_extra.py
    C = np.array(json.load(open(RES / "ldavis_figure5_data.json"))["corr"])
    cmap = LinearSegmentedColormap.from_list("div", ["#104281", "#6da7ec", EMPTY, "#ec8f8e", "#a32c2c"])
    # off-diagonal lower triangle only: rows T2-T3, columns T1-T2
    S = C[1:, :2]
    mask = np.triu(np.ones_like(S, bool), 1)
    fig, ax = plt.subplots(figsize=(W * 0.6, W * 0.32))
    im = ax.imshow(np.ma.array(S, mask=mask), cmap=cmap, vmin=-1, vmax=1)
    for i in range(2):
        for j in range(2):
            if not mask[i, j]:
                v = S[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5, color=INK,
                        fontweight="bold" if v == S[~mask].min() else "normal")
    ax.set_xticks(range(2)); ax.set_xticklabels([f"T{t}" for t in (1, 2)], fontsize=7.5)
    ax.set_yticks(range(2)); ax.set_yticklabels([f"T{t} · {TNAME[t]}" for t in (2, 3)], fontsize=6.8)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[-1, -0.5, 0, 0.5, 1]); cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=2)
    save(fig, "fig06_topic_correlation.png")

# ---------- Figures 7 and 8: authors, sources ----------
def name_variants(item_name):
    """Ways the same author can be written in the 'author' field ('Ana Souza', 'Souza, Ana', 'Souza, A.', 'A. Souza')."""
    pieces = item_name.split()
    first_name, surname = pieces[0], " ".join(pieces[1:]) if len(pieces) > 2 and pieces[1][0].islower() else pieces[-1]
    if len(pieces) > 2 and not pieces[1][0].islower():
        surname = pieces[-1]
    start_ = first_name[0]
    return [rf"\b{re.escape(item_name)}\b", rf"\b{re.escape(surname)},\s*{re.escape(first_name)}\b",
            rf"{re.escape(surname)},\s*{start_}\.", rf"\b{start_}\.\s*{re.escape(surname)}"]

def authors_by_topic(au, d):
    """Full-name variants first; initials only when they reproduce the count of Top_Authors
    (initials alone over-count common names such as 'Y. Li')."""
    lines = []
    for item_name, expected in zip(au["Author"], au["Documents"]):
        v = name_variants(str(item_name))
        for variant_set in (v[:2], v):
            pad = "|".join(f"(?:{x})" for x in variant_set)
            sel = d[d["author"].fillna("").str.contains(pad, case=False, regex=True)]
            if len(sel) == expected:
                break
        lines.append({"Author": item_name, **{t: int((sel["main_topic"] == t).sum()) for t in TOPICS}})
    return pd.DataFrame(lines)

def fig7_8():
    b = pd.read_excel(RES / "Bibliographic Analysis/BIBLIOGRAPHIC_ANALYSIS.xlsx", sheet_name=None)
    d = docs_with_topics()
    # Figure 7: bubble matrix, author x topic (bubble area = documents)
    au = b["Top_Authors"].head(15)
    m = authors_by_topic(au, d)
    m["Total"] = m[list(TOPICS)].sum(axis=1)
    dif = (m["Total"] != au["Documents"].values).sum()
    print(f"   authors: counts differ from Top_Authors for {dif} of {len(m)}")
    fig, ax = plt.subplots(figsize=(W, 4.4))
    yy = np.arange(len(m))[::-1]
    for t in TOPICS:
        for y, v in zip(yy, m[t]):
            if v:
                ax.scatter(t, y, s=110 * v, color=TOPIC[t], edgecolor=SURF, linewidth=1.2, zorder=3)
                ax.text(t, y, v, ha="center", va="center", fontsize=6.8, color="white", fontweight="bold", zorder=4)
    for y, v in zip(yy, m["Total"]):
        ax.text(4.0, y, v, ha="center", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(yy); ax.set_yticklabels([clean(a) for a in m["Author"]], fontsize=7.2)
    tick_labels = ["T%d\n%s" % (t, TNAME[t].replace(" & ", " &\n", 1)) for t in TOPICS] + ["Total"]
    ax.set_xticks([1, 2, 3, 4]); ax.set_xticklabels(tick_labels, fontsize=6.5)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    for y in yy: ax.axhline(y, color=GRID, lw=0.5, zorder=1)
    ax.axvline(3.55, color=GRID, lw=0.8)
    ax.set_xlim(0.4, 4.4); ax.set_ylim(-0.8, len(m) - 0.2)
    save(fig, "fig07_top_authors.png")

    # Figure 8: three-field Sankey, author keywords -> LDA topics -> main sources (matplotlib)
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch

    synonyms = {"ai": "artificial intelligence", "digitalization": "digital transformation",
                 "digitalisation": "digital transformation", "supply chain": "supply chain management",
                 "decision-making": "decision making", "lean manufacturing": "lean production",
                 "industry4.0": "industry 4.0", "covid19": "covid-19"}
    exclude_terms = {"china", "india", "africa", "philippines", "brazil", "case study", "framework", "review",
               "literature review", "systematic literature review"}
    def keyword_set(k):
        out = set()
        for w in re.split(r"[;,]", str(k)):
            w = re.sub(r"\s+", " ", w.strip().lower())
            w = synonyms.get(w, w)
            if w and w not in exclude_terms:
                out.add(w)
        return out
    d["kw"] = d["keywords"].apply(lambda k: keyword_set(k) if pd.notna(k) else set())
    freq = pd.Series([w for s in d["kw"] for w in s]).value_counts()
    kws = freq.head(15).index.tolist()
    sources = d["journal"].value_counts().head(10).index.tolist()

    left_flows = {(w, t): int(((d["main_topic"] == t) & d["kw"].apply(lambda s: w in s)).sum())
           for w in kws for t in TOPICS}
    right_flows = {(t, j): int(((d["main_topic"] == t) & (d["journal"] == j)).sum()) for j in sources for t in TOPICS}
    dominant = lambda pairs, key: max(TOPICS, key=lambda t: pairs[key(t)])
    kws.sort(key=lambda w: (dominant(left_flows, lambda t: (w, t)), -freq[w]))
    sources.sort(key=lambda j: (dominant(right_flows, lambda t: (t, j)), -sum(right_flows[(t, j)] for t in TOPICS)))

    tot_w = {w: sum(left_flows[(w, t)] for t in TOPICS) for w in kws}
    tot_j = {j: sum(right_flows[(t, j)] for t in TOPICS) for j in sources}
    inflow = {t: sum(left_flows[(w, t)] for w in kws) for t in TOPICS}
    outflow = {t: sum(right_flows[(t, j)] for j in sources) for t in TOPICS}
    topic_height = {t: max(inflow[t], outflow[t]) for t in TOPICS}

    gap = 0.012
    scale = (1 - gap * 5 * 3) / sum(topic_height.values())       # one scale for all three columns
    xs, wn, wm = (0.0, 0.5, 1.0), 0.018, 0.04

    def column_layout(block_items, heights, gap_y):
        total = sum(heights[i] * scale for i in block_items) + gap_y * (len(block_items) - 1)
        y, pos = 0.5 + total / 2, {}
        for i in block_items:
            h = heights[i] * scale
            pos[i] = [y, y, y]                          # topo, cursor esquerdo, cursor direito
            y -= h + gap_y
        return pos

    pw, pt, pj = column_layout(kws, tot_w, gap * 0.9), column_layout(list(TOPICS), topic_height, gap * 3), \
        column_layout(sources, tot_j, gap * 1.2)

    def flow_band(x0, a0, h1, x1, b0, h2, colour):
        xm = (x0 + x1) / 2
        verts = [(x0, a0), (xm, a0), (xm, b0), (x1, b0), (x1, b0 - h2), (xm, b0 - h2), (xm, a0 - h1),
                 (x0, a0 - h1), (x0, a0)]
        codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CURVE4,
                 MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts, codes), fc=colour, ec=SURF, lw=0.3, alpha=0.45))

    fig, ax = plt.subplots(figsize=(W, 5.2)); ax.axis("off")
    for w in kws:                                        # keyword -> topic
        for t in TOPICS:
            v = left_flows[(w, t)]
            if v:
                h = v * scale
                a0 = pw[w][1]; pw[w][1] -= h
                b0 = pt[t][1]; pt[t][1] -= h
                flow_band(xs[0] + wn, a0, h, xs[1] - wm / 2, b0, h, TOPIC[t])
    for t in TOPICS:                                # topic -> source
        for j in sources:
            v = right_flows[(t, j)]
            if v:
                h = v * scale
                a0 = pt[t][2]; pt[t][2] -= h
                b0 = pj[j][2]; pj[j][2] -= h
                flow_band(xs[1] + wm / 2, a0, h, xs[2] - wn, b0, h, TOPIC[t])
    for w in kws:
        h = tot_w[w] * scale
        ax.add_patch(Rectangle((xs[0], pw[w][0] - h), wn, h, color=MUTED, lw=0))
        ax.text(xs[0] - 0.01, pw[w][0] - h / 2, f"{w} ({freq[w]})", ha="right", va="center", fontsize=6.8)
    for t in TOPICS:
        h = topic_height[t] * scale
        ax.add_patch(Rectangle((xs[1] - wm / 2, pt[t][0] - h), wm, h, color=TOPIC[t], lw=0))
        ax.text(xs[1], pt[t][0] - h / 2, f"T{t}", ha="center", va="center", fontsize=7.5, color="white",
                fontweight="bold", rotation=90)
    for j in sources:
        h = tot_j[j] * scale
        ax.add_patch(Rectangle((xs[2] - wn, pj[j][0] - h), wn, h, color=MUTED, lw=0))
        ax.text(xs[2] + 0.01, pj[j][0] - h / 2, f"{clean(j, 48)} ({tot_j[j]})", ha="left", va="center",
                fontsize=6.8)
    for x, label in ((xs[0], "Author keywords"), (xs[1], "LDA topics"), (xs[2], "Main sources")):
        ax.text(x + (wn / 2 if x != xs[1] else 0), 1.035, label, ha="center", va="bottom", fontsize=7.5,
                fontweight="bold", color=INK2)
    hand = [plt.Line2D([0], [0], marker="s", ls="", ms=7, color=TOPIC[t]) for t in TOPICS]
    ax.legend(hand, [f"T{t} · {TNAME[t]}" for t in TOPICS], loc="upper center", bbox_to_anchor=(0.5, 0.0),
              ncol=2, frameon=False, fontsize=7, handletextpad=0.3, columnspacing=1.2)
    ax.set_xlim(-0.42, 1.56); ax.set_ylim(-0.02, 1.07)
    save(fig, "fig08_top_sources.png")
    print("   keywords:", {w: tot_w[w] for w in kws})
    print("   keyword -> topic:", inflow, "| topic -> source:", outflow)

# ---------- Figure 9: topics by year ----------
def fig9():
    t = pd.read_excel(RES / "Bibliographic Analysis/BIBLIOGRAPHIC_ANALYSIS.xlsx", sheet_name="Topics_by_Year")
    t.columns = [str(c) for c in t.columns]
    years = np.arange(int(t["Year"].min()), int(t["Year"].max()) + 1)
    t = t.set_index("Year").reindex(years, fill_value=0)
    fig, ax = plt.subplots(figsize=(W, 2.9))
    for k in TOPICS:
        s = t[str(k)]
        ax.plot(years, s, color=TOPIC[k], lw=2 if k == 1 else 1.6, zorder=3 if k == 1 else 2,
                solid_capstyle="round")
        ax.plot(2025, s.loc[2025], "o", ms=4.5, color=TOPIC[k], mec=SURF, mew=1, zorder=4)
    ends = {k: int(t[str(k)].loc[2025]) for k in TOPICS}
    order = sorted(ends, key=ends.get)
    ypos, last = {}, -99
    for k in order:
        ypos[k] = max(ends[k], last + 8); last = ypos[k]
    for k in TOPICS:
        ax.text(years[-1] + 1.0, ypos[k], f"T{k} · {TNAME[k]} ({ends[k]} in 2025)", fontsize=6.8, color=INK2, va="center")
    ax.axvspan(2025.5, years[-1] + 0.5, color=EMPTY, zorder=0)
    ax.text(2026.5, 121, "2026–2027 (partial)", fontsize=6.3, color=INK2, ha="center", va="center")
    ax.set_xlim(years[0] - 0.5, years[-1] + 18); ax.set_ylim(0, 125)
    ax.set_xticks([y for y in years if y % 5 == 0]); ax.set_ylabel("Articles per year"); grid(ax, "y")
    ax.spines["bottom"].set_bounds(years[0] - 0.5, years[-1] + 0.5)
    save(fig, "fig09_topics_by_year.png")

# ---------- Figure 10: Hoshin matrix ----------
CATS = [("Organizational\nCulture", 1, 5), ("Capa-\nbilities", 6, 7), ("Focus", 8, 10),
        ("Alignment", 11, 14), ("Integration", 15, 19), ("Review", 20, 20)]
def fig10():
    m = pd.read_excel(RES / "Hoshin Kanri/ALIGNMENT_MATRIX_FILLED.xlsx")
    g = [f"D{i:02d}" for i in range(1, 21)]
    top = m.sort_values(["Total", "Rank"], ascending=[False, True]).head(40).reset_index(drop=True)
    X = top[g].values
    fig, ax = plt.subplots(figsize=(W, 8.2))
    ax.imshow(X, cmap=ListedColormap([EMPTY, TOPIC[1]]), aspect="auto", vmin=0, vmax=1, interpolation="nearest")
    for x in np.arange(-0.5, 20, 1): ax.axvline(x, color=SURF, lw=1.2)
    for y in np.arange(-0.5, 40, 1): ax.axhline(y, color=SURF, lw=1.2)
    for _, a, b in CATS[1:]:
        ax.axvline(a - 1.5, color=INK2, lw=1)
    for name, a, b in CATS:
        ax.text((a + b) / 2 - 1, -1.2, name, ha="center", va="bottom", fontsize=6.8, fontweight="bold", color=INK, linespacing=1.1)
    labels = [f"{int(r.Year)} · {clean(r.Title, 62)}" for r in top.itertuples()]
    ax.set_yticks(range(len(top))); ax.set_yticklabels(labels, fontsize=6.2)
    ax.set_xticks(range(20)); ax.set_xticklabels(g, fontsize=6.8, rotation=90)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    for i, v in enumerate(top["Total"]):
        ax.text(20.0, i, str(int(v)), va="center", fontsize=6.5, color=INK2)
    ax.text(20.0, -1.2, "n", fontsize=6.8, fontweight="bold", va="bottom")
    ax.set_xlim(-0.5, 20.6)
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=fig.transFigure, fill=False, lw=0))
    h1 = plt.Rectangle((0, 0), 1, 1, fc=TOPIC[1]); h0 = plt.Rectangle((0, 0), 1, 1, fc=EMPTY, ec="#cfcec9", lw=0.5)
    ax.legend([h1, h0], ["expression identified", "not identified"], loc="upper left", bbox_to_anchor=(0, -0.045),
              ncol=2, frameon=False, fontsize=7, handlelength=1.2, handleheight=1.0)
    save(fig, "fig10_hoshin_matrix.png")

# ---------- Figure 11: all 20 guidelines ----------
def fig11():
    c = pd.read_excel(RES / "Hoshin Kanri/table_guidelines_complete.xlsx").set_index("Guideline")
    short = {"D01": "Continuous improvement culture", "D02": "Challenging goals", "D03": "Active leadership involvement",
             "D04": "Nemawashi", "D05": "Systemic view", "D06": "Problem-solving and teamwork",
             "D07": "Cross-functional teams", "D08": "Meaningful long-term vision", "D09": "Measurable medium-term goals",
             "D10": "Vital few annual policies", "D11": "Policies deployed at each level", "D12": "Participation in annual planning",
             "D13": "Catchball", "D14": "Cause-and-effect analysis", "D15": "Goals in management routines",
             "D16": "Responsibility via cascade", "D17": "Team self-discipline", "D18": "Periodic performance reports",
             "D19": "Visible, updated information", "D20": "Top-management diagnosis"}
    g = [f"D{i:02d}" for i in range(1, 21)]
    vals = [c.loc[d, "Percentage"] for d in g]
    ns = [int(c.loc[d, "Articles"]) for d in g]
    fig, ax = plt.subplots(figsize=(W, 4.6))
    y = np.arange(20)[::-1]
    ax.barh(y, vals, height=0.62, color=TOPIC[1], edgecolor=SURF, linewidth=1)
    for yi, v, n in zip(y, vals, ns):
        ax.text(v + 0.08, yi, f"{v:.1f}% ({n})" if n else "0", va="center", fontsize=7, color=INK2 if n else MUTED)
    ax.set_yticks(y); ax.set_yticklabels([f"{d}  {short[d]}" for d in g], fontsize=7)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False); grid(ax, "x")
    for name, a, b in CATS:
        y0, y1 = 20 - b - 0.4, 20 - a + 0.4
        ax.text(9.15, (y0 + y1) / 2, name.replace("-\n", "").replace("\n", " "), ha="right", va="center", fontsize=7,
                fontweight="bold", color=INK2)
        if a > 1: ax.axhline(20 - a + 0.5, color=MUTED, lw=0.6)
    ax.set_xlim(0, 9.2); ax.set_xticks(range(0, 8)); ax.set_xlabel("Topic 1 articles with dictionary mentions (%), n = 502, reference run")
    save(fig, "fig11_guidelines.png")

# ---------- Record selection flow (v16) ----------
def fig_flow():
    """PRISMA-style flow of the records, from the database search to the Topic 1 alignment analysis."""
    fig, ax = plt.subplots(figsize=(W, 5.0)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    def box(x, y, w, h, text, fc=SURF, ec="#86b6ef", fs=7, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
                fontweight="bold" if bold else "normal", linespacing=1.25)
    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9,
                                                             shrinkA=0, shrinkB=0, mutation_scale=8))
    for y, name in [(84, "Identification"), (56, "Screening"), (22, "Analysis")]:
        ax.text(1, y, name, rotation=90, ha="left", va="center", fontsize=7.5, fontweight="bold", color=INK2)
    box(6, 82, 54, 14, "Records retrieved by the search string (n = 1,849)\nScienceDirect 1,778 · Scopus 57\n"
        "IEEE Xplore 9 · Web of Science 5")
    box(66, 82, 32, 14, "Removed by the document-type filter\nof each database before export\n(n = 562)",
        fc=EMPTY, ec="#b9b8b2", fs=6.6)
    arrow(60.6, 89, 65.4, 89)
    box(6, 61, 54, 13, "Research articles exported (n = 1,287)\nScienceDirect 1,256 · Scopus 28\n"
        "Web of Science 2 · IEEE Xplore 1")
    arrow(33, 81.4, 33, 73.6)
    box(66, 62, 32, 11, "Duplicates removed\n(normalised title; n = 15)", fc=EMPTY, ec="#b9b8b2", fs=6.6)
    arrow(60.6, 67.5, 65.4, 67.5)
    box(6, 44, 54, 11, "Unique articles in the corpus (n = 1,272)")
    arrow(33, 61.4, 33, 55.6)
    box(66, 44, 32, 11, "No abstract or abstract\n< 100 characters (n = 23)", fc=EMPTY, ec="#b9b8b2", fs=6.6)
    arrow(60.6, 49.5, 65.4, 49.5)
    box(6, 28, 54, 9, "Documents in the LDA model (n = 1,249)")
    arrow(33, 43.4, 33, 37.6)
    tw = 17
    for i, (t, n) in enumerate([(1, 502), (2, 430), (3, 317)]):
        x = 6 + i * (tw + 1.5)
        box(x, 10, tw, 9, f"Topic {t}\n(n = {n})", fc="#eef5fd" if t == 1 else SURF,
            ec=FOCAL if t == 1 else "#b9b8b2", fs=6.8, bold=t == 1)
        arrow(33, 27.4, x + tw / 2, 19.6)
    ax.text(50, 3.5, "Dominant topic of each document. Topic 1 (n = 502) was used in the guideline alignment analysis;\n"
            "the guideline expressions were also counted in the 1,272 and 1,249 documents (sensitivity check).",
            ha="center", va="center", fontsize=6.5, color=INK2)
    save(fig, "fig_flow_selection.png")

for f in (fig1, fig_flow, fig2, fig3, fig4, fig5, fig6, fig7_8, fig9, fig10, fig11):
    f()
