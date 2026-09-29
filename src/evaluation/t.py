# -*- conding: utf-8 -*-
# @Time    : 2026/4/5  16:23
# @Author  : psi

import json
from rdkit import Chem
from rdkit.Chem import Draw
import pubchempy as pcp
import requests
from pathlib import Path


def read_file(file):

    data = json.load(open(file, 'r', encoding='utf-8'))

    print(1)


# =========================
# SMILES → Image
# =========================
def smiles_to_image1(smiles, size=(300, 300)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)


def smiles_to_pubchem_2d_image(smiles: str, save_path: str) -> int:
    """
    根据 SMILES 从 PubChem 查询化合物，并保存 2D 结构图（PNG）

    参数
    ----
    smiles : str
        化合物的 SMILES 表示
    save_path : str
        图片保存路径（如 "./benzaldehyde_2d.png"）

    返回
    ----
    cid : int
        PubChem CID
    """
    # 1. 通过 SMILES 查询 PubChem
    compounds = pcp.get_compounds(smiles, namespace="smiles")
    if not compounds:
        raise ValueError(f"未在 PubChem 中找到对应 SMILES: {smiles}")

    compound = compounds[0]
    cid = compound.cid

    # 2. 构造 2D 结构图下载 URL
    img_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG"

    # 3. 下载图片
    response = requests.get(img_url, timeout=30)
    response.raise_for_status()

    # 4. 保存图片
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f:
        f.write(response.content)

    return cid


def get_best_structure_image(smiles, save_path):
    try:
        # 尝试从 PubChem 获取高质量官方图
        cid = smiles_to_pubchem_2d_image(smiles, save_path)
        print(f"Success: Image from PubChem (CID: {cid})")
    except Exception as e:
        # 如果查不到或断网，则本地渲染
        img = smiles_to_image1(smiles)
        if img:
            img.save(save_path)
            print("Fallback: Image rendered by RDKit")
        else:
            print("Error: Invalid SMILES")


if __name__ == '__main__':

    read_file('./data/CASMI2016_Cat2and3_Challenge_1-pos.instrument_fields_smiles.json')

    print(1)


