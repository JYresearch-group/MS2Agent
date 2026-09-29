import json, csv, os, random, re
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdFMCS
import numpy as np

BASE = r'F:\数据库多智能体'

print("=" * 60, flush=True)
print("STEP 1: Load SMILES", flush=True)
print("=" * 60, flush=True)

# Train SMILES (streaming)
train_set = set()
chunk = ''
with open(os.path.join(BASE, 'train_12290.json'), 'r', encoding='utf-8') as f:
    while True:
        piece = f.read(5 * 1024 * 1024)
        if not piece: break
        chunk += piece
        for m in re.finditer(r'"smiles"\s*:\s*"([^"]{8,})"', chunk):
            train_set.add(m.group(1))
        chunk = chunk[-500:]
train_all = sorted(train_set)
print(f"  Train unique SMILES: {len(train_all)}", flush=True)

# CASMI 2022 SMILES (497 unique from res_report CSV)
casmi_set = set()
csv_path = os.path.join(BASE, 'MS2agent', 'res_report_anthropic_claude-sonnet-4-6_v2.csv')
with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
    reader = csv.DictReader(f)
    for row in reader:
        s = row.get('smiles', '')
        if s: casmi_set.add(s)
casmi_all = sorted(casmi_set)
print(f"  CASMI 2022 unique SMILES: {len(casmi_all)}", flush=True)

# Generate fingerprints
print("\nGenerating fingerprints...", flush=True)
def get_fps(smiles_list):
    valid = []
    fps = []
    for sm in smiles_list:
        mol = Chem.MolFromSmiles(sm)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            valid.append((sm, mol, mol.GetNumBonds()))
            fps.append(fp)
    return valid, fps

train_data, train_fps = get_fps(train_all)
casmi_data, casmi_fps = get_fps(casmi_all)
print(f"  Train valid: {len(train_fps)}, CASMI valid: {len(casmi_fps)}", flush=True)

# ============================================================
# STEP 2: Tanimoto Similarity (all pairs)
# ============================================================
print(f"\n{'=' * 60}", flush=True)
print(f"STEP 2: Tanimoto - {len(train_fps)} x {len(casmi_fps)} = {len(train_fps)*len(casmi_fps):,} pairs", flush=True)
print(f"{'=' * 60}", flush=True)

all_sims = []
start_t = __import__('time').time()
for i, fp in enumerate(train_fps):
    sims = DataStructs.BulkTanimotoSimilarity(fp, casmi_fps)
    all_sims.extend(sims)
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(train_fps)}, {__import__('time').time()-start_t:.1f}s", flush=True)

tani_arr = np.array(all_sims, dtype=np.float32)
print(f"\nDone in {__import__('time').time()-start_t:.1f}s", flush=True)
print(f"  Min={tani_arr.min():.4f}, Max={tani_arr.max():.4f}", flush=True)
print(f"  Mean={tani_arr.mean():.4f}, Median={np.median(tani_arr):.4f}", flush=True)

# Tanimoto histogram
bins = np.linspace(0, 1, 21)
hist, edges = np.histogram(tani_arr, bins=bins)
max_h = max(hist) if max(hist) > 0 else 1
print(f"\n  Tanimoto Distribution:", flush=True)
for i in range(len(hist)):
    if hist[i] > 0:
        bar = '#' * max(1, int(hist[i] * 40 / max_h))
        pct = hist[i] / len(tani_arr) * 100
        print(f"    [{edges[i]:.2f},{edges[i+1]:.2f}): {hist[i]:8,d} ({pct:5.1f}%) {bar}", flush=True)

# Per-train-molecule max Tanimoto
print(f"\n  Per-train-molecule max Tanimoto:", flush=True)
train_max_tani = []
for i, fp in enumerate(train_fps):
    sims = DataStructs.BulkTanimotoSimilarity(fp, casmi_fps)
    train_max_tani.append(max(sims))
train_max_arr = np.array(train_max_tani)
print(f"    Min={train_max_arr.min():.4f}, Max={train_max_arr.max():.4f}", flush=True)
print(f"    Mean={train_max_arr.mean():.4f}, Median={np.median(train_max_arr):.4f}", flush=True)
print(f"    Train molecules with max_Tani>=0.5: {int((train_max_arr>=0.5).sum())}/{len(train_max_arr)}", flush=True)
print(f"    Train molecules with max_Tani>=0.7: {int((train_max_arr>=0.7).sum())}/{len(train_max_arr)}", flush=True)

# Per-casmi-molecule max Tanimoto
casmi_max_tani = []
for j, fp in enumerate(casmi_fps):
    sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
    casmi_max_tani.append(max(sims))
casmi_max_arr = np.array(casmi_max_tani)
print(f"\n  Per-CASMI-molecule max Tanimoto (vs all train):", flush=True)
print(f"    Min={casmi_max_arr.min():.4f}, Max={casmi_max_arr.max():.4f}", flush=True)
print(f"    Mean={casmi_max_arr.mean():.4f}, Median={np.median(casmi_max_arr):.4f}", flush=True)
print(f"    CASMI molecules with max_Tani>=0.5: {int((casmi_max_arr>=0.5).sum())}/{len(casmi_max_arr)}", flush=True)
print(f"    CASMI molecules with max_Tani>=0.7: {int((casmi_max_arr>=0.7).sum())}/{len(casmi_max_arr)}", flush=True)

# ============================================================
# STEP 3: MCES (sampled pairs)
# ============================================================
print(f"\n{'=' * 60}", flush=True)
print("STEP 3: MCES (sampled)", flush=True)
print(f"{'=' * 60}", flush=True)

# Sample pairs across Tanimoto ranges
random.seed(42)
n_per_bin = 30
mces_results = []

# Get all pair indices with their Tanimoto
print("  Building pair list for sampling...", flush=True)
pair_tani = []
for i, fp in enumerate(train_fps):
    sims = DataStructs.BulkTanimotoSimilarity(fp, casmi_fps)
    for j, s in enumerate(sims):
        pair_tani.append((i, j, s))

# Sample from different Tanimoto ranges
tani_bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]
sampled_pairs = []
for lo, hi in tani_bins:
    bucket = [(i, j, t) for i, j, t in pair_tani if lo <= t < hi]
    if bucket:
        sample = random.sample(bucket, min(n_per_bin, len(bucket)))
        sampled_pairs.extend(sample)

print(f"  Total sampled pairs: {len(sampled_pairs)}", flush=True)

# Compute MCES for sampled pairs
print("  Computing MCES...", flush=True)
for idx, (ti, ci, t) in enumerate(sampled_pairs):
    sm1, mol1, nb1 = train_data[ti]
    sm2, mol2, nb2 = casmi_data[ci]
    try:
        mcs = rdFMCS.FindMCS([mol1, mol2], timeout=3,
            bondCompare=rdFMCS.BondCompare.CompareAny,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            ringMatchesRingOnly=True, completeRingsOnly=False)
        Ec = 0 if (mcs.canceled or mcs.numBonds == 0) else mcs.numBonds
        mces = nb1 + nb2 - 2 * Ec
        mces_results.append((t, mces, nb1, nb2, Ec))
    except:
        pass
    if (idx + 1) % 30 == 0:
        print(f"    {idx+1}/{len(sampled_pairs)}", flush=True)

print(f"\n  MCES computed: {len(mces_results)}", flush=True)

# MCES by Tanimoto bin
print(f"\n  {'Tanimoto':<15} {'N':>5} {'MCES_mean':>10} {'MCES_med':>10} {'MCES_range':>12}", flush=True)
for lo, hi in tani_bins:
    bucket = [(t, m) for t, m, _, _, _ in mces_results if lo <= t < hi]
    if bucket:
        mces_vals = [m for _, m in bucket]
        print(f"  [{lo:.1f},{hi:.1f})       {len(bucket):5d} {np.mean(mces_vals):10.1f} {np.median(mces_vals):10.1f} [{min(mces_vals):3d}-{max(mces_vals):3d}]", flush=True)
    else:
        print(f"  [{lo:.1f},{hi:.1f})       {0:5d} {'N/A':>10} {'N/A':>10}", flush=True)

# Linear fit
if len(mces_results) > 10:
    t_arr = np.array([r[0] for r in mces_results])
    m_arr = np.array([r[1] for r in mces_results])
    corr = np.corrcoef(t_arr, m_arr)[0, 1]
    coeffs = np.polyfit(t_arr, m_arr, 1)
    print(f"\n  Correlation: {corr:.4f}", flush=True)
    print(f"  Linear fit: MCES ≈ {coeffs[0]:.1f} * Tanimoto + {coeffs[1]:.1f}", flush=True)

# Save
output = {
    'train_unique': len(train_all),
    'casmi_unique': len(casmi_all),
    'tanimoto': {
        'n_pairs': len(tani_arr),
        'min': float(tani_arr.min()), 'max': float(tani_arr.max()),
        'mean': float(tani_arr.mean()), 'median': float(np.median(tani_arr)),
    },
    'per_train_max_tanimoto': {
        'min': float(train_max_arr.min()), 'max': float(train_max_arr.max()),
        'mean': float(train_max_arr.mean()), 'median': float(np.median(train_max_arr)),
        'ge_0.5': int((train_max_arr >= 0.5).sum()),
        'ge_0.7': int((train_max_arr >= 0.7).sum()),
    },
    'per_casmi_max_tanimoto': {
        'min': float(casmi_max_arr.min()), 'max': float(casmi_max_arr.max()),
        'mean': float(casmi_max_arr.mean()), 'median': float(np.median(casmi_max_arr)),
        'ge_0.5': int((casmi_max_arr >= 0.5).sum()),
        'ge_0.7': int((casmi_max_arr >= 0.7).sum()),
    },
    'mces_sampled': len(mces_results),
}
with open(os.path.join(BASE, 'tanimoto_mces_train_vs_casmi2022.json'), 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: tanimoto_mces_train_vs_casmi2022.json", flush=True)
print("DONE!", flush=True)
