import json
import sys
from rdkit import Chem
from rdkit.Chem import rdFMCS
from collections import Counter

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

data_dir = sys.argv[1] if len(sys.argv) > 1 else '.'

with open(data_dir + '/gym_out_sample_id_4295.json', 'r') as f:
    gym = json.load(f)

total = len(gym)
mces_distances = []
not_computed = 0

for idx, item in enumerate(gym):
    gt_smiles = item['ms_smiles']['smiles']
    top1_smiles = item['res_candidates'][0][0]

    mol_gt = Chem.MolFromSmiles(gt_smiles)
    mol_pred = Chem.MolFromSmiles(top1_smiles)

    if mol_gt is None or mol_pred is None:
        not_computed += 1
        continue

    try:
        res = rdFMCS.FindMCS([mol_gt, mol_pred])
        if res.numAtoms == 0:
            dist = mol_gt.GetNumBonds() + mol_pred.GetNumBonds()
        else:
            dist = mol_gt.GetNumBonds() + mol_pred.GetNumBonds() - 2 * res.numBonds
        mces_distances.append(dist)
    except:
        not_computed += 1

print(f'Mass-based MCES@1 (top-1 candidate vs ground truth)')
print(f'Computed: {len(mces_distances)}/{total}')
print(f'Not computed: {not_computed}')
print(f'Mean MCES@1: {sum(mces_distances)/len(mces_distances):.2f}')
print(f'Median: {sorted(mces_distances)[len(mces_distances)//2]}')
print(f'Min: {min(mces_distances)}, Max: {max(mces_distances)}')

dist_bins = Counter()
for d in mces_distances:
    if d == 0: dist_bins['0'] += 1
    elif d <= 3: dist_bins['1-3'] += 1
    elif d <= 6: dist_bins['4-6'] += 1
    elif d <= 10: dist_bins['7-10'] += 1
    elif d <= 20: dist_bins['11-20'] += 1
    else: dist_bins['>20'] += 1

print(f'\nDistribution:')
for k in ['0', '1-3', '4-6', '7-10', '11-20', '>20']:
    cnt = dist_bins.get(k, 0)
    print(f'  MCES dist {k}: {cnt} ({cnt/len(mces_distances)*100:.1f}%)')
