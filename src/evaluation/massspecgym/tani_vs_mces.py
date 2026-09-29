import json, os, random, re
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdFMCS
import numpy as np

BASE = r'F:\数据库多智能体'

# 加载 SMILES
train_smiles_set = set()
chunk = ''
with open(os.path.join(BASE, 'train_12290.json'), 'r', encoding='utf-8') as f:
    while True:
        piece = f.read(5*1024*1024)
        if not piece: break
        chunk += piece
        for m in re.finditer(r'"smiles"\s*:\s*"([^"]{8,})"', chunk):
            train_smiles_set.add(m.group(1))
        chunk = chunk[-500:]

with open(os.path.join(BASE, 'gym_out_sample_id_4295.json'), 'r', encoding='utf-8') as f:
    gym = json.load(f)
gym_set = set()
for e in gym:
    s = e['ms_smiles'].get('smiles', '')
    if s: gym_set.add(s)

with open(os.path.join(BASE, 'massspecgym_test_3170_smiles.txt'), 'r', encoding='utf-8') as f:
    msg_list = [l.strip() for l in f if l.strip()]

random.seed(42)
train_sample = random.sample(sorted(train_smiles_set), 100)

def parse_with_fp(smiles_list):
    mols = []
    fps = []
    for sm in smiles_list:
        mol = Chem.MolFromSmiles(sm)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            mols.append((sm, mol, mol.GetNumBonds()))
            fps.append(fp)
    return mols, fps

train_m, train_fps = parse_with_fp(train_sample)
gym_m, gym_fps = parse_with_fp(sorted(gym_set))
msg_m, msg_fps = parse_with_fp(msg_list)

print(f"Train: {len(train_m)}, Gym: {len(gym_m)}, MSG: {len(msg_m)}", flush=True)

# 批量 Tanimoto，找 Tanimoto ≈ 0.35 的对
print("\nFinding pairs with Tanimoto in [0.25, 0.45]...", flush=True)
target_pairs = []
all_tani = []

for i, (sm1, mol1, nb1) in enumerate(train_m):
    sims = DataStructs.BulkTanimotoSimilarity(train_fps[i], gym_fps)
    for j, sim in enumerate(sims):
        all_tani.append(sim)
        if 0.25 <= sim <= 0.45:
            target_pairs.append((train_m[i], gym_m[j], sim))

    sims2 = DataStructs.BulkTanimotoSimilarity(train_fps[i], msg_fps)
    for j, sim in enumerate(sims2):
        all_tani.append(sim)
        if 0.25 <= sim <= 0.45:
            target_pairs.append((train_m[i], msg_m[j], sim))

print(f"Total pairs: {len(all_tani)}, Target pairs: {len(target_pairs)}", flush=True)

# 对目标对计算 MCES
print("\nComputing MCES for target pairs...", flush=True)
results = []
for idx, ((sm1, mol1, nb1), (sm2, mol2, nb2), tani) in enumerate(target_pairs):
    try:
        mcs = rdFMCS.FindMCS([mol1, mol2], timeout=3,
            bondCompare=rdFMCS.BondCompare.CompareAny,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            ringMatchesRingOnly=True, completeRingsOnly=False)
        Ec = 0 if (mcs.canceled or mcs.numBonds == 0) else mcs.numBonds
        mces = nb1 + nb2 - 2 * Ec
        results.append((tani, mces, nb1, nb2, Ec))
    except:
        pass
    if (idx+1) % 50 == 0:
        print(f"  {idx+1}/{len(target_pairs)}", flush=True)

print(f"MCES computed: {len(results)}", flush=True)

# 分桶统计
print("\n=== Tanimoto vs MCES ===", flush=True)
print(f"{'Tanimoto':<15} {'N':>6} {'MCES_mean':>10} {'MCES_med':>10} {'MCES_range':>15}", flush=True)

bins = [(0.00, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.25),
        (0.25, 0.30), (0.30, 0.35), (0.35, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 1.01)]

for lo, hi in bins:
    bucket = [(t, m) for t, m, _, _, _ in results if lo <= t < hi]
    if bucket:
        mces_vals = [m for _, m in bucket]
        print(f"[{lo:.2f},{hi:.2f})       {len(bucket):6d} {np.mean(mces_vals):10.1f} {np.median(mces_vals):10.1f} [{min(mces_vals):3d}-{max(mces_vals):3d}]", flush=True)
    else:
        print(f"[{lo:.2f},{hi:.2f})       {0:6d} {'N/A':>10} {'N/A':>10}", flush=True)

# Tanimoto ≈ 0.35
print(f"\n=== Tanimoto ≈ 0.35 (0.33-0.37) ===", flush=True)
near = [(t, m, nb1, nb2, ec) for t, m, nb1, nb2, ec in results if 0.33 <= t <= 0.37]
if near:
    mces_vals = [r[1] for r in near]
    print(f"样本数: {len(near)}", flush=True)
    print(f"MCES 均值: {np.mean(mces_vals):.1f}", flush=True)
    print(f"MCES 中位数: {np.median(mces_vals):.1f}", flush=True)
    print(f"MCES 范围: [{min(mces_vals)} - {max(mces_vals)}]", flush=True)
    for t, m, nb1, nb2, ec in sorted(near)[:5]:
        print(f"  Tani={t:.3f} MCES={m:3d} (|E1|={nb1}, |E2|={nb2}, |Ec|={ec})", flush=True)

# 整体相关性
if results:
    tani_arr = np.array([r[0] for r in results])
    mces_arr = np.array([r[1] for r in results])
    corr = np.corrcoef(tani_arr, mces_arr)[0, 1]
    print(f"\n相关系数: {corr:.4f}", flush=True)
    coeffs = np.polyfit(tani_arr, mces_arr, 1)
    print(f"线性拟合: MCES ≈ {coeffs[0]:.1f} * Tanimoto + {coeffs[1]:.1f}", flush=True)
    print(f"Tanimoto=0.35 → MCES ≈ {coeffs[0]*0.35 + coeffs[1]:.1f}", flush=True)
