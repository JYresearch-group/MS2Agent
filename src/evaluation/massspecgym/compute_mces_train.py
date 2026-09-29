import json, time, os, sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from rdkit import Chem
from rdkit.Chem import rdFMCS
import numpy as np

print("Loading data...")

# Train SMILES
with open('F:/数据库多智能体/train_12290.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)
train_smiles_set = set()
for entry in train_data:
    nr = entry.get('normalized_record', {})
    sm = nr.get('smiles', '')
    if not sm:
        for key in ['combined_summary_text', 'structure_summary_text']:
            txt = entry.get(key, '')
            if 'smiles:' in txt:
                for line in txt.split('\n'):
                    if line.strip().startswith('smiles:'):
                        sm = line.strip().split('smiles:', 1)[1].strip()
                        break
            if sm: break
    if sm: train_smiles_set.add(sm)
train_smiles = sorted(train_smiles_set)
print(f"Train: {len(train_data)} entries -> {len(train_smiles)} unique SMILES")

# Gym 967
with open('F:/数据库多智能体/gym_out_sample_id_4295.json', 'r', encoding='utf-8') as f:
    gym = json.load(f)
gym_smiles_set = set()
for e in gym:
    s = e['ms_smiles'].get('smiles', '')
    if s: gym_smiles_set.add(s)
gym_smiles = sorted(gym_smiles_set)
print(f"Gym: {len(gym_smiles)} unique SMILES")

# MSG 3170
with open('F:/数据库多智能体/massspecgym_test_3170_smiles.txt', 'r', encoding='utf-8') as f:
    msg_smiles = [l.strip() for l in f if l.strip()]
print(f"MSG: {len(msg_smiles)} SMILES")

# 解析为 (smiles, n_bonds) 列表，预计算所有分子的 RDKit Mol
def parse_all(smiles_list, name):
    parsed = []
    fail = 0
    for sm in smiles_list:
        mol = Chem.MolFromSmiles(sm)
        if mol:
            parsed.append((sm, mol, mol.GetNumBonds()))
        else:
            fail += 1
    print(f"  {name}: {len(parsed)} OK, {fail} fail")
    return parsed

print("Parsing molecules...")
train_parsed = parse_all(train_smiles, "Train")
gym_parsed = parse_all(gym_smiles, "Gym")
msg_parsed = parse_all(msg_smiles, "MSG")

# MCES 计算
def compute_mces(mol1, mol2):
    try:
        mcs = rdFMCS.FindMCS(
            [mol1, mol2], timeout=5,
            bondCompare=rdFMCS.BondCompare.CompareAny,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            ringMatchesRingOnly=True, completeRingsOnly=False
        )
        if mcs.canceled or mcs.numBonds == 0:
            Ec = 0
        else:
            Ec = mcs.numBonds
        return mol1.GetNumBonds() + mol2.GetNumBonds() - 2 * Ec
    except:
        return None

def batch_mces(train_list, test_list, name):
    n = len(train_list) * len(test_list)
    print(f"\n{'='*60}")
    print(f"{name}: {len(train_list)} x {len(test_list)} = {n:,} pairs")
    print(f"{'='*60}")

    results = []
    start = time.time()
    pair_count = 0

    for sm1, mol1, nb1 in train_list:
        for sm2, mol2, nb2 in test_list:
            d = compute_mces(mol1, mol2)
            if d is not None:
                results.append(d)
            pair_count += 1
            if pair_count % 200000 == 0:
                elapsed = time.time() - start
                rate = pair_count / elapsed
                remaining = (n - pair_count) / rate / 60
                print(f"  {pair_count:,}/{n:,} ({pair_count/n*100:.1f}%), {len(results)} ok, {elapsed:.0f}s, ETA {remaining:.0f}min")

    arr = np.array(results, dtype=np.float32)
    elapsed = time.time() - start
    print(f"Done! {len(arr):,}/{n:,} in {elapsed:.0f}s")
    print(f"  Min={int(arr.min())}, Max={int(arr.max())}")
    print(f"  Mean={arr.mean():.2f}, Median={np.median(arr):.1f}, Std={arr.std():.2f}")

    # 分桶
    print(f"\n  Distribution:")
    bins = list(range(0, int(arr.max())+6, 4))
    hist, edges = np.histogram(arr, bins=bins)
    for i in range(len(hist)):
        if hist[i] > 0:
            bar = '#' * (hist[i] // (max(hist)//30 or 1))
            pct = hist[i]/len(arr)*100
            print(f"    [{edges[i]:3d}, {edges[i+1]:3d}): {hist[i]:7,d} ({pct:5.1f}%) {bar}")

    return arr

gym_arr = batch_mces(train_parsed, gym_parsed, "Train vs Gym (967)")
msg_arr = batch_mces(train_parsed, msg_parsed, "Train vs MSG (3170)")

# 保存
with open('F:/数据库多智能体/mces_train_vs_both.json', 'w', encoding='utf-8') as f:
    json.dump({
        'train_unique': len(train_smiles),
        'gym_unique': len(gym_smiles),
        'msg_unique': len(msg_smiles),
        'train_vs_gym': {
            'total_pairs': int(len(gym_arr)),
            'min': int(gym_arr.min()), 'max': int(gym_arr.max()),
            'mean': float(gym_arr.mean()), 'median': float(np.median(gym_arr)),
            'std': float(gym_arr.std()),
            'values': gym_arr.tolist()
        },
        'train_vs_msg': {
            'total_pairs': int(len(msg_arr)),
            'min': int(msg_arr.min()), 'max': int(msg_arr.max()),
            'mean': float(msg_arr.mean()), 'median': float(np.median(msg_arr)),
            'std': float(msg_arr.std()),
            'values': msg_arr.tolist()
        }
    }, f, ensure_ascii=False)
print(f"\nSaved: mces_train_vs_both.json")
