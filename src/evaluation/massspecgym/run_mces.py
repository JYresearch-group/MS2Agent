import json, time, os, random
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from rdkit import Chem
from rdkit.Chem import rdFMCS, AllChem, DataStructs
import numpy as np
import re

BASE = r'F:\数据库多智能体'
OUT_TANI = os.path.join(BASE, 'tanimoto_train_vs_both.json')
OUT_MCES = os.path.join(BASE, 'mces_30_vs_both.json')

print("Step 1: Extract SMILES (streaming)...", flush=True)
train_smiles_set = set()
chunk = ''
with open(os.path.join(BASE, 'train_12290.json'), 'r', encoding='utf-8') as f:
    while True:
        piece = f.read(1024 * 1024 * 5)
        if not piece: break
        chunk += piece
        for m in re.finditer(r'"smiles"\s*:\s*"([^"]{8,})"', chunk):
            train_smiles_set.add(m.group(1))
        chunk = chunk[-1000:]
train_all = sorted(train_smiles_set)
print(f"  Train unique: {len(train_all)}", flush=True)

with open(os.path.join(BASE, 'gym_out_sample_id_4295.json'), 'r', encoding='utf-8') as f:
    gym = json.load(f)
gym_smiles_set = set()
for e in gym:
    s = e['ms_smiles'].get('smiles', '')
    if s: gym_smiles_set.add(s)
gym_all = sorted(gym_smiles_set)
print(f"  Gym unique: {len(gym_all)}", flush=True)

with open(os.path.join(BASE, 'massspecgym_test_3170_smiles.txt'), 'r', encoding='utf-8') as f:
    msg_smiles = [l.strip() for l in f if l.strip()]
print(f"  MSG: {len(msg_smiles)}", flush=True)

# ============================================================
# PART A: Tanimoto similarity (FAST - full computation)
# ============================================================
print("\nStep 2: Tanimoto similarity (fast)...", flush=True)

def to_fp(sm):
    mol = Chem.MolFromSmiles(sm)
    if mol:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return None

random.seed(42)
n_sample = min(300, len(train_all))
train_sample = random.sample(train_all, n_sample)

train_fps = [(sm, to_fp(sm)) for sm in train_sample]
train_fps = [(sm, fp) for sm, fp in train_fps if fp]
gym_fps = [(sm, to_fp(sm)) for sm in gym_all]
gym_fps = [(sm, fp) for sm, fp in gym_fps if fp]
msg_fps = [(sm, to_fp(sm)) for sm in msg_smiles]
msg_fps = [(sm, fp) for sm, fp in msg_fps if fp]

print(f"  Train FP: {len(train_fps)}, Gym FP: {len(gym_fps)}, MSG FP: {len(msg_fps)}", flush=True)

def tanimoto_batch(train_list, test_list, name):
    n = len(train_list) * len(test_list)
    print(f"\n  {name}: {len(train_list)} x {len(test_list)} = {n:,} pairs", flush=True)
    results = []
    start = time.time()
    cnt = 0
    for sm1, fp1 in train_list:
        sims = []
        for sm2, fp2 in test_list:
            sims.append(DataStructs.TanimotoSimilarity(fp1, fp2))
            cnt += 1
        results.extend(sims)
        if cnt % 100000 == 0:
            print(f"    {cnt:,}/{n:,} ({cnt/n*100:.1f}%), {time.time()-start:.0f}s", flush=True)
    arr = np.array(results, dtype=np.float32)
    print(f"  Done: {len(arr):,} in {time.time()-start:.0f}s", flush=True)
    print(f"  Min={arr.min():.4f}, Max={arr.max():.4f}, Mean={arr.mean():.4f}, Median={np.median(arr):.4f}", flush=True)

    # Histogram
    bins = np.linspace(0, 1, 21)
    hist, edges = np.histogram(arr, bins=bins)
    max_h = max(hist) if max(hist) > 0 else 1
    for i in range(len(hist)):
        if hist[i] > 0:
            bar = '#' * max(1, hist[i] * 30 // max_h)
            pct = hist[i]/len(arr)*100
            print(f"    [{edges[i]:.2f},{edges[i+1]:.2f}): {hist[i]:7,d} ({pct:5.1f}%) {bar}", flush=True)
    return arr.tolist()

tani_gym = tanimoto_batch(train_fps, gym_fps, "Tanimoto: Train(300) vs Gym(967)")
tani_msg = tanimoto_batch(train_fps, msg_fps, "Tanimoto: Train(300) vs MSG(3170)")

# Save Tanimoto results
with open(OUT_TANI, 'w', encoding='utf-8') as f:
    json.dump({
        'sampled_train': n_sample,
        'gym_unique': len(gym_all),
        'msg_unique': len(msg_smiles),
        'train_vs_gym': {
            'n': len(tani_gym), 'min': min(tani_gym), 'max': max(tani_gym),
            'mean': float(np.mean(tani_gym)), 'median': float(np.median(tani_gym)),
            'values': tani_gym
        },
        'train_vs_msg': {
            'n': len(tani_msg), 'min': min(tani_msg), 'max': max(tani_msg),
            'mean': float(np.mean(tani_msg)), 'median': float(np.median(tani_msg)),
            'values': tani_msg
        }
    }, f)
print(f"\nTanimoto saved: {OUT_TANI}", flush=True)

# ============================================================
# PART B: MCES (small sample, short timeout)
# ============================================================
print("\nStep 3: MCES (30 molecules, timeout=1s)...", flush=True)

random.seed(42)
n_mces = min(30, len(train_all))
train_mces_sample = random.sample(train_all, n_mces)

def parse_mols(smiles_list):
    result = []
    for sm in smiles_list:
        mol = Chem.MolFromSmiles(sm)
        if mol: result.append((mol, mol.GetNumBonds()))
    return result

train_m = parse_mols(train_mces_sample)
gym_m = parse_mols(gym_all)
msg_m = parse_mols(msg_smiles)
print(f"  Parsed: train={len(train_m)}, gym={len(gym_m)}, msg={len(msg_m)}", flush=True)

def mces(mol1, mol2):
    try:
        r = rdFMCS.FindMCS([mol1, mol2], timeout=1,
            bondCompare=rdFMCS.BondCompare.CompareAny,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            ringMatchesRingOnly=True, completeRingsOnly=False)
        Ec = 0 if (r.canceled or r.numBonds == 0) else r.numBonds
        return mol1.GetNumBonds() + mol2.GetNumBonds() - 2 * Ec
    except: return None

def mces_batch(train_list, test_list, name):
    n = len(train_list) * len(test_list)
    print(f"\n  {name}: {len(train_list)} x {len(test_list)} = {n:,} pairs", flush=True)
    results = []
    start = time.time()
    cnt = 0
    for mol1, nb1 in train_list:
        for mol2, nb2 in test_list:
            d = mces(mol1, mol2)
            if d is not None: results.append(d)
            cnt += 1
            if cnt % 20000 == 0:
                print(f"    {cnt:,}/{n:,} ({cnt/n*100:.1f}%), {len(results)} ok, {time.time()-start:.0f}s", flush=True)
    arr = np.array(results, dtype=np.float32)
    print(f"  Done: {len(arr):,} in {time.time()-start:.0f}s", flush=True)
    print(f"  Min={int(arr.min())}, Max={int(arr.max())}, Mean={arr.mean():.2f}, Median={np.median(arr):.1f}", flush=True)

    bins = list(range(0, int(arr.max())+6, 4))
    hist, edges = np.histogram(arr, bins=bins)
    max_h = max(hist) if max(hist) > 0 else 1
    for i in range(len(hist)):
        if hist[i] > 0:
            bar = '#' * max(1, hist[i] * 30 // max_h)
            pct = hist[i]/len(arr)*100
            print(f"    [{edges[i]:3d},{edges[i+1]:3d}): {hist[i]:7,d} ({pct:5.1f}%) {bar}", flush=True)
    return arr.tolist()

mces_gym = mces_batch(train_m, gym_m, "MCES: Train(30) vs Gym(967)")
mces_msg = mces_batch(train_m, msg_m, "MCES: Train(30) vs MSG(3170)")

with open(OUT_MCES, 'w', encoding='utf-8') as f:
    json.dump({
        'sampled_train': n_mces,
        'train_vs_gym': {'n': len(mces_gym), 'min': min(mces_gym), 'max': max(mces_gym),
            'mean': float(np.mean(mces_gym)), 'median': float(np.median(mces_gym)), 'values': mces_gym},
        'train_vs_msg': {'n': len(mces_msg), 'min': min(mces_msg), 'max': max(mces_msg),
            'mean': float(np.mean(mces_msg)), 'median': float(np.median(mces_msg)), 'values': mces_msg}
    }, f)
print(f"\nMCES saved: {OUT_MCES}", flush=True)
print("\nALL DONE!", flush=True)
