import json
from collections import Counter
from pathlib import Path

r = json.load(open("evaluation/results/latest_results.json", encoding="utf-8"))
c = Counter(f"{rec['model']}:{rec.get('group', rec.get('scenario_id','')[:1])}" for rec in r)
groups = Counter(f"{rec['model']}:{rec['ground_truth'].get('conflict_expected','')}" for rec in r[:10])
# Show scenario groups covered
sc_groups = Counter(f"{rec['model']}" for rec in r)
print("Records per model:", dict(sc_groups))
grp_counts = Counter(f"{rec['model']}:{rec.get('group','')}" for rec in r)
for k, v in sorted(grp_counts.items()):
    print(f"  {k}: {v}")
