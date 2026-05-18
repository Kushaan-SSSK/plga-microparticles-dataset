"""
_package_submission_fixes.py
Copies all submission deliverables into submission_fixes/ folder.
"""
import shutil, os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dest = os.path.join(base, 'submission_fixes')

copies = [
    ('scripts/regenerate_fig5_burst_audit.py',        'scripts/regenerate_fig5_burst_audit.py'),
    ('scripts/run_loso_restricted.py',                'scripts/run_loso_restricted.py'),
    ('outputs/burst24h_scaling_audit.csv',            'outputs/burst24h_scaling_audit.csv'),
    ('outputs/burst24h_scaling_summary.csv',          'outputs/burst24h_scaling_summary.csv'),
    ('outputs/burst24h_class_counts_after_audit.csv', 'outputs/burst24h_class_counts_after_audit.csv'),
    ('outputs/loso_restricted_summary.csv',           'outputs/loso_restricted_summary.csv'),
    ('figures/Fig_5.png',                             'figures/Fig_5.png'),
    ('elsarticle-revised.tex',                        'manuscript/elsarticle-revised.tex'),
]

print(f"Packaging into: {dest}\n")
for rel_src, rel_dst in copies:
    src = os.path.join(base, *rel_src.split('/'))
    dst = os.path.join(dest, *rel_dst.split('/'))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    kb = round(os.path.getsize(dst) / 1024, 1)
    print(f'  OK  {rel_dst:<55} {kb:6.1f} KB')

print('\nDone.')
