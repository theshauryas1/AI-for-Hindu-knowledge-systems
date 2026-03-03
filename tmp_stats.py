import json, sys
from pathlib import Path
sys.path.insert(0, '.')
results = json.load(open('evaluation/results/_merged_for_stats.json', encoding='utf-8'))
from evaluation.eval_metrics import bootstrap_confidence_interval, mcnemar_test, cohens_kappa, compute_error_taxonomy

models = ['vanilla_llm','rag','symbolic','hybrid']

print('BOOTSTRAP CI (partial_acc, n=2000):')
for m in models:
    ci = bootstrap_confidence_interval(results, m, n_bootstrap=2000, metric='partial_acc')
    if 'error' not in ci:
        print(f'  {m:16}  mean={ci["mean"]:.3f}  95%CI=[{ci["ci_lower"]:.3f}, {ci["ci_upper"]:.3f}]  n={ci["n_scenarios"]}')
    else:
        print(f'  {m:16}  {ci["error"]}')

print()
print('McNEMAR (hybrid vs others):')
pairs=[('hybrid','symbolic'),('hybrid','vanilla_llm'),('hybrid','rag')]
for a,b in pairs:
    r = mcnemar_test(results, a, b)
    if 'error' not in r:
        sig = "SIG*" if r["significant_p05"] else "not-sig"
        print(f'  {a} vs {b}: chi2={r["chi2"]:.3f} p={r["p_value"]:.4f} [{sig}] - {r["interpretation"]}')
    else:
        print(f'  {a} vs {b}: {r["error"]}')

print()
print('COHENS KAPPA:')
for a,b in [('hybrid','symbolic'),('hybrid','vanilla_llm'),('symbolic','vanilla_llm')]:
    r = cohens_kappa(results, a, b)
    if 'error' not in r:
        print(f'  {a} vs {b}: kappa={r["kappa"]:.3f} ({r["interpretation"]}) n_pairs={r["n_pairs"]}')
    else:
        print(f'  {a} vs {b}: {r["error"]}')

print()
print('ERROR TAXONOMY (hybrid):')
tax = compute_error_taxonomy(results)
hyb_tax = tax['by_model'].get('hybrid', {})
total = sum(hyb_tax.values())
for cat, cnt in sorted(hyb_tax.items(), key=lambda x: -x[1]):
    pct = 100*cnt/max(total,1)
    print(f'  {cat:30} {cnt:4}  ({pct:.0f}%)')

print()
print('ERROR TAXONOMY BY SCHOOL (hybrid errors):')
for school,d in tax['by_school'].items():
    total2 = sum(d.values())
    errs = [f'{k}:{v}' for k,v in sorted(d.items(), key=lambda x: -x[1]) if k!='correct']
    print(f'  {school:20} total={total2} | {", ".join(errs[:3])}')
