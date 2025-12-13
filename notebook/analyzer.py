#!/usr/bin/env python3
import json, glob, os, math, re
from collections import defaultdict, Counter
import pandas as pd
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
IN_PATTERN = "./*.json"      # folder with your user JSON files
OUT_DIR = "analysis_out"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- HELPERS ----------
def safe_first(x, default=None):
    return x[0] if isinstance(x, list) and x else default

def norm_path(p):
    if not isinstance(p, str): return p
    return re.sub(r"^/data/upload/\d+/", "", p)  # strip Label Studio prefix for cross-file matching

def task_key(task_obj):
    """
    Stable image key across users. Prefer explicit id.
    Fallback to uploaded filename or data.image.
    """
    tid = task_obj.get("id")
    if tid is not None:
        return f"id:{tid}"
    fn = task_obj.get("file_upload") or task_obj.get("data", {}).get("image")
    return f"file:{norm_path(fn)}"

def extract_label(result):
    """
    Works for Label Studio choices classification.
    Returns e.g. '1_Pronacio' / '2_Neutralis' / '3_Szupinacio'.
    """
    if not result: 
        return None
    val = result[0].get("value", {})
    ch = val.get("choices", [])
    return safe_first(ch)

# ---------- LOAD ----------
per_task = defaultdict(lambda: {"labels": {}, "meta": {}})
label_counts = Counter()

files = glob.glob(IN_PATTERN)
if not files:
    raise SystemExit("No JSON files matched. Adjust IN_PATTERN.")

for fp in files:
    user = os.path.splitext(os.path.basename(fp))[0]  # filename = annotator
    try:
        data = json.load(open(fp, "r", encoding="utf-8"))
    except Exception as e:
        print(f"Skip {fp}: {e}")
        continue

    # Normalize export shapes
    if isinstance(data, dict):
        data = data.get("tasks") or data.get("results") or data.get("data") or []
    if not isinstance(data, list):
        print(f"Skip {fp}: unsupported JSON shape")
        continue

    for task in data:
        k = task_key(task)
        img = task.get("file_upload") or task.get("data", {}).get("image")
        if img: per_task[k]["meta"]["image"] = norm_path(img)

        anns = task.get("annotations") or []
        if not anns:
            continue
        # Take the last completed result in that file as this user's label for the task
        lbl = None
        for ann in anns:
            cand = extract_label(ann.get("result"))
            if cand: lbl = cand
        if not lbl:
            continue

        per_task[k]["labels"][user] = lbl
        label_counts[lbl] += 1

# ---------- CONSENSUS PER IMAGE ----------
rows = []
for k, node in per_task.items():
    labels_by_user = node["labels"]
    if not labels_by_user:
        continue
    n = len(labels_by_user)
    freq = Counter(labels_by_user.values())
    top_label, top_count = (None, 0) if not freq else freq.most_common(1)[0]
    agreement = top_count / n if n else math.nan
    tie = sum(1 for c in freq.values() if c == top_count) > 1
    consensus = "TIE" if tie else top_label
    rows.append({
        "task_key": k,
        "image": node["meta"].get("image"),
        "n_annotators": n,
        "consensus_label": consensus,
        "agreement_ratio": round(agreement, 4),
        "label_hist": dict(freq)
    })
consensus_df = pd.DataFrame(rows).sort_values(["agreement_ratio","task_key"], ascending=[False, True])
consensus_df.to_csv(os.path.join(OUT_DIR, "consensus_per_image.csv"), index=False)

# ---------- PER-USER ACCURACY VS CONSENSUS ----------
acc = defaultdict(lambda: {"match":0, "total":0})
per_user_label_counts = Counter()

for tk, node in per_task.items():
    gold = next((r["consensus_label"] for r in [dict(consensus_df.loc[consensus_df["task_key"]==tk].iloc[0])] ), None)
    if gold in (None, "TIE"):
        continue
    for user, lbl in node["labels"].items():
        acc[user]["total"] += 1
        if lbl == gold:
            acc[user]["match"] += 1
        per_user_label_counts[(user, lbl)] += 1

acc_rows = []
for user, d in sorted(acc.items()):
    total = d["total"]
    match = d["match"]
    acc_rows.append({
        "annotator": user,
        "matches": match,
        "total": total,
        "accuracy": round(match/total, 4) if total else None
    })
acc_df = pd.DataFrame(acc_rows).sort_values(["accuracy","annotator"], ascending=[False, True])
acc_df.to_csv(os.path.join(OUT_DIR, "annotator_accuracy_vs_consensus.csv"), index=False)

# ---------- OVERALL AND PER-USER DISTRIBUTIONS ----------
lbl_df = pd.DataFrame(
    [{"label": k, "count": v} for k, v in sorted(label_counts.items())]
).sort_values("count", ascending=False)
lbl_df["percent"] = (lbl_df["count"] / lbl_df["count"].sum() * 100).round(2)
lbl_df.to_csv(os.path.join(OUT_DIR, "label_distribution_overall.csv"), index=False)

per_user_rows = [{"annotator": u, "label": l, "count": c}
                 for (u,l), c in per_user_label_counts.items()]
per_user_df = pd.DataFrame(per_user_rows).sort_values(["annotator","label"])
per_user_df.to_csv(os.path.join(OUT_DIR, "label_distribution_per_user.csv"), index=False)

# ---------- CHARTS ----------
plt.figure()
plt.bar(lbl_df["label"], lbl_df["count"])
plt.title("Label counts (overall)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "label_counts_overall.png"), dpi=200)
plt.close()

# Agreement histogram
agreements = consensus_df["agreement_ratio"].value_counts().sort_index()
plt.figure()
plt.bar(agreements.index.astype(str), agreements.values)
plt.title("Consensus agreement ratio frequency")
plt.xlabel("Agreement ratio")
plt.ylabel("Images")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "consensus_agreement_hist.png"), dpi=200)
plt.close()

# ---------- SUMMARY ----------
print(f"Users detected: {len(set(u for _, node in per_task.items() for u in node['labels'].keys()))}")
print(f"Images with annotations: {len(consensus_df)}")
print("Outputs written to:", os.path.abspath(OUT_DIR))
