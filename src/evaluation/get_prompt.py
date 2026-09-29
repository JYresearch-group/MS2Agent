# -*- conding: utf-8 -*-
# @Time    : 2026/4/5  17:01
# @Author  : psi


import base64
import requests
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw
import pubchempy as pcp


# ================= 1. 质谱数据预处理 =================
def preprocess_ms_peaks(ms_list, top_n=15):
    """筛选强度前 N 的特征峰，并按 m/z 升序排列"""
    # 按强度降序
    sorted_by_intensity = sorted(ms_list, key=lambda x: x, reverse=True)
    # 截取前 N 个
    top_peaks = sorted_by_intensity[:top_n]
    # 按 m/z 升序排列（符合解谱习惯）
    return sorted(top_peaks, key=lambda x: x)


# ================= 2. 图像获取逻辑 =================
def get_structure_image(smiles, save_path="molecule.png"):
    """
    策略：优先从 PubChem 获取官方图，失败则使用 RDKit 本地渲染
    """
    path = Path(save_path)
    # 尝试 PubChem
    try:
        compounds = pcp.get_compounds(smiles, namespace="smiles")
        if compounds:
            cid = compounds.cid
            img_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG"
            response = requests.get(img_url, timeout=10)
            if response.status_code == 200:
                with open(path, "wb") as f:
                    f.write(response.content)
                return str(path), "PubChem Official"
    except Exception:
        pass

    # RDKit 备选方案
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        Draw.MolToFile(mol, str(path), size=(400, 400))
        return str(path), "RDKit Rendered"

    return None, None


def image_to_base64(image_path):
    """将图片转换为 base64 编码供 LLM 调用"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


# ================= 3. 自动化分析流程 =================
def generate_analysis_payload(data):
    # A. 提取并清洗数据
    filtered_ms = preprocess_ms_peaks(data['ms'], top_n=15)
    img_path, img_source = get_structure_image(data['smiles'])

    # B. 格式化质谱峰列表字符串
    ms_string = "\n".join([f"- m/z: {p:.4f}, Intensity: {p:.1f}" for p in filtered_ms])

    # C. 构建第一个 Prompt：仪器数据分析
    instrument_prompt = f"""
### Task: 质谱仪器数据深度分析
**背景信息:**
- 仪器类型: {data['Instrument']}
- 离子源/模式: {data['Ion_Source']} / {data['Ion_Mode']}
- 母离子 m/z: {data['parent_mz']} (Charge: {data['Charge']})
- 扫描号: {data['scan']}

**主要特征峰 (Top 15 by Intensity):**
{ms_string}

**分析维度:**
1. **质量偏差分析:** 结合 {data['Instrument']} 的精度，评估母离子与理论质量的契合度。
2. **离子化机理:** 解释在 {data['Ion_Source']} {data['Ion_Mode']} 模式下分子的带电原理。
3. **碎片模式分析:** 识别基峰，并计算主要碎片与母离子的质量差（Neutral Loss），推测可能的化学键断裂位置。
"""

    # D. 构建第二个 Prompt：SMILES + 图像多模态分析
    img_base64 = image_to_base64(img_path) if img_path else None

    structure_prompt = f"""
### Task: 化学结构与图像综合解读 (基于 {img_source} 图像)
**输入 SMILES:** {data['smiles']}

**分析维度:**
1. **结构校验:** 验证图像中的拓扑连接与 SMILES 描述是否完全一致。
2. **空间特征:** 通过图像评估分子的空间拥挤度（位阻）及对称性。
3. **官能团活性:** 识别关键官能团，并结合图像指出分子的电子分布特征（如亲电位点）。
4. **质谱关联预测:** 在图像中标注出在 CID 碰撞中预计最不稳定的共价键，并关联质谱数据中的主要碎片。
"""

    return instrument_prompt, structure_prompt, img_base64


# --- 模拟 JSON 数据输入 ---
raw_data = {
    "parent_mz": "167.104",
    "scan": "1",
    "Instrument": "Orbitrap",
    "Ion_Source": "LC-ESI",
    "Ion_Mode": "Positive",
    "Charge": "1",
    "smiles": "NC1=NC(NC2CC2)=NC(N)=N1",
    "ms": [[56.049, 890015], [85.050, 32494888], [167.103, 206684544], [60.055, 19158144], [125.082, 17335174]]  # 示例缩略
}

# 运行
ins_p, str_p, img_b64 = generate_analysis_payload(raw_data)

print("--- [Prompt 1: 仪器分析] ---")
print(ins_p)
print("\n--- [Prompt 2: 结构分析] ---")
print(str_p)



