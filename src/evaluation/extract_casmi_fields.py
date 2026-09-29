import argparse
import json
from pathlib import Path

try:
    from rdkit import Chem
except ImportError:
    Chem = None
from tqdm import tqdm


# DEFAULT_SOURCE_PATH = Path("./data/CASMI2016_Cat2and3_Challenge_1-pos.json")
# DEFAULT_SOURCE_PATH = Path('/data0/xj/Slime_MS/data/all_data_list_pos_neg_v5_new_filter_14.json')
DEFAULT_SOURCE_PATH = Path("./data/casmi2022_all_in_one_structured.json")


def to_int(value):
    if value in (None, ""):
        return None
    # 处理列表类型，取第一个元素
    if isinstance(value, list):
        if len(value) == 0:
            return None
        value = value[0]
        return to_int(value)  # 递归处理
    # 处理字符串如 'MS2' -> 2
    if isinstance(value, str):
        value = value.strip().upper()
        # 尝试提取数字部分
        import re
        match = re.search(r'(\d+)', value)
        if match:
            return int(match.group(1))
        # 尝试直接转换
        try:
            return int(value)
        except ValueError:
            return None
    # 处理数字类型
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def to_float(value):
    if value in (None, ""):
        return None
    # 处理列表类型，取第一个元素
    if isinstance(value, list):
        if len(value) == 0:
            return None
        return to_float(value[0])  # 递归处理
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def is_valid_smiles(smiles):
    """使用 RDKit 验证 SMILES 是否有效"""
    if not smiles or not isinstance(smiles, str):
        return False
    smiles = smiles.strip()
    if not smiles:
        return False
    if Chem is None:
        # 如果没有 RDKit，只进行基本检查
        return True
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def is_valid_ms(ms):
    """验证 MS 数据是否有效"""
    if not ms:
        return False
    if isinstance(ms, list):
        return len(ms) > 0
    if isinstance(ms, str):
        return ms.strip() != ""
    return True


def get_precursor_mz(json_data, smiles=None):
    # float(json_data.get("precursor_mz", 0.0))
    if 'precursor_mz' in json_data:
        return float(json_data["precursor_mz"])
    elif 'parent_mz' in json_data:
        return float(json_data["parent_mz"])
    elif 'precursormz' in json_data:
        return float(json_data["precursormz"])
    elif 'emass' in json_data:
        return float(json_data["emass"])
    elif 'q1' in json_data:
        return float(json_data["q1"])
    elif 'exactmass' in json_data and json_data["exactmass"] is not None:
        return float(json_data["exactmass"])
    # elif 'quantmass' in json_data and json_data["quantmass"] is not None:
    #     return float(json_data["quantmass"])

    else:
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        # smiles = "CC1CCC2CCCCC2C1=O"  # 类似你这个分子
        mol = Chem.MolFromSmiles(smiles)

        exact_mass = Descriptors.ExactMolWt(mol)
        # print(exact_mass)

        return float(exact_mass)


def get_ms(json_data):

    if "mz_lists" in json_data:
        ms_peaks = json_data.get("mz_lists", [])
    # elif "mz_list" in data:
    #     ms_peaks = json_data.get("mz_list", [])
    elif "ms_list" in json_data:
        ms_peaks = json_data.get("ms_list", [])
    else:
        ms_peaks = json_data.get("ms", [])

    return ms_peaks


def get_scan(json_data):
    """提取 scan 值，尝试多种可能的键名"""
    if 'scan' in json_data:
        return json_data["scan"]
    elif 'scannum' in json_data:
        return json_data["scannum"]
    elif 'scan_num' in json_data:
        return json_data["scan_num"]
    elif 'scanNumber' in json_data:
        return json_data["scanNumber"]
    elif 'scan_number' in json_data:
        return json_data["scan_number"]
    else:
        return None


def get_ms_level(json_data):
    """提取 ms_level 值，尝试多种可能的键名"""
    if 'ms_level' in json_data:
        return json_data["ms_level"]
    elif 'mslevel' in json_data:
        return json_data["mslevel"]
    elif 'MSLevel' in json_data:
        return json_data["MSLevel"]
    elif 'msLevel' in json_data:
        return json_data["msLevel"]
    elif 'level' in json_data:
        return json_data["level"]
    else:
        return None


def get_ion_mode(json_data):
    """提取 ion_mode 值，尝试多种可能的键名"""
    if 'ionmode' in json_data:
        return json_data["ionmode"]
    elif 'ion_mode' in json_data:
        return json_data["ion_mode"]
    elif 'IonMode' in json_data:
        return json_data["IonMode"]
    elif 'ionMode' in json_data:
        return json_data["ionMode"]
    elif 'mode' in json_data:
        return json_data["mode"]
    elif 'ionization_mode' in json_data:
        return json_data["ionization_mode"]
    else:
        return None


def get_ion_source(json_data):
    """提取 ion_source 值，尝试多种可能的键名"""
    if 'source' in json_data:
        return json_data["source"]
    elif 'ion_source' in json_data:
        return json_data["ion_source"]
    elif 'ionSource' in json_data:
        return json_data["ionSource"]
    elif 'IonSource' in json_data:
        return json_data["IonSource"]
    elif 'ionization_source' in json_data:
        return json_data["ionization_source"]
    else:
        return None


def get_instrument(json_data):
    """提取 instrument 值，尝试多种可能的键名"""
    if 'instrument_type' in json_data:
        return json_data["instrument_type"]
    elif 'instrument' in json_data:
        return json_data["instrument"]
    elif 'Instrument' in json_data:
        return json_data["Instrument"]
    elif 'instrument_name' in json_data:
        return json_data["instrument_name"]
    elif 'analyzer' in json_data:
        return json_data["analyzer"]
    else:
        return None


def get_charge(json_data):
    """提取 charge 值，尝试多种可能的键名"""
    if 'charge' in json_data:
        return json_data["charge"]
    elif 'Charge' in json_data:
        return json_data["Charge"]
    elif 'precursor_charge' in json_data:
        return json_data["precursor_charge"]
    elif 'ion_charge' in json_data:
        return json_data["ion_charge"]
    elif 'polarity' in json_data:
        # 有时极性用 +1/-1 或正/负表示
        polarity = json_data["polarity"]
        if polarity in [1, '1', 'positive', 'POS', 'pos', 'Pos', 'Positive']:
            return 1
        elif polarity in [-1, '-1', 'negative', 'NEG', 'neg', 'Neg', 'Negative']:
            return -1
        else:
            return polarity
    else:
        return None


def get_adduct(json_data):
    """提取 adduct 值，尝试多种可能的键名"""
    if 'adduct' in json_data:
        return json_data["adduct"]


def normalize_record(record):
    record1 = {k.lower(): v for k, v in record.items()}
    record.update(record1)
    smiles = record.get("smiles", None)

    ms = get_ms(record)

    # 验证 SMILES
    if not is_valid_smiles(smiles):
        return None

    # 验证 MS 数据
    if not is_valid_ms(ms):
        return None

    # 使用解析函数获取各字段值
    scan = get_scan(record)
    ms_level = get_ms_level(record)
    ion_source = get_ion_source(record)
    instrument = get_instrument(record)
    charge = get_charge(record)
    ion_mode = get_ion_mode(record)
    adduct = get_adduct(record)

    return {
        "ms": ms,
        "parent_mz": to_float(get_precursor_mz(record, smiles)),
        "scan": to_int(scan),
        "ms_level": to_int(ms_level),
        "Ion_Mode": ion_mode.strip() if isinstance(ion_mode, str) else ion_mode,
        "Ion_Source": ion_source,
        "Instrument": instrument,
        "Charge": to_int(charge),
        "smiles": smiles,
        'adduct': adduct,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract instrument-related fields and SMILES from a CASMI JSON file."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_SOURCE_PATH),
        help="Input CASMI JSON file.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output JSON file. Defaults to <input>.instrument_fields_smiles.json",
    )
    return parser.parse_args()


def infer_output_path(input_path: Path) -> Path:
    if input_path.suffix == ".json":
        return input_path.with_name(f"{input_path.stem}.instrument_fields_smiles.json")
    return input_path.with_name(f"{input_path.name}.instrument_fields_smiles.json")


def make_hashable(val):
    """将值转换为可哈希类型用于统计"""
    if val is None:
        return None
    if isinstance(val, (list, dict)):
        return str(val)[:50]  # 转为字符串并限制长度
    return val


def analyze_fields(records):
    """统计各字段的分布情况"""
    from collections import Counter

    stats = {
        "scan": Counter(),
        "ms_level": Counter(),
        "ion_mode": Counter(),
        "ion_source": Counter(),
        "instrument": Counter(),
        "charge": Counter(),
    }

    for record in tqdm(records, desc='Analyzing fields'):
        stats["scan"][make_hashable(get_scan(record))] += 1
        stats["ms_level"][make_hashable(get_ms_level(record))] += 1
        stats["ion_mode"][make_hashable(get_ion_mode(record))] += 1
        stats["ion_source"][make_hashable(get_ion_source(record))] += 1
        stats["instrument"][make_hashable(get_instrument(record))] += 1
        stats["charge"][make_hashable(get_charge(record))] += 1

    print("\n=== 字段统计 ===")
    for field, counter in stats.items():
        print(f"\n{field}:")
        total = sum(counter.values())
        for val, count in counter.most_common():
            pct = count / total * 100
            print(f"  {val!r}: {count} ({pct:.1f}%)")

    return stats


def main():
    args = parse_args()
    source_path = Path(args.input)
    output_path = Path(args.output) if args.output else infer_output_path(source_path)

    with source_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    total_count = len(records)

    # 先统计字段分布
    analyze_fields(records)

    extracted = []
    invalid_smiles_count = 0
    invalid_ms_count = 0

    for record in tqdm(records, desc='Processing records'):
        normalized = normalize_record(record)
        if normalized is None:
            # 判断是哪种无效
            # smiles = record.get("smiles")
            # ms = record.get("ms")
            # if not is_valid_smiles(smiles):
            #     invalid_smiles_count += 1
            # elif not is_valid_ms(ms):
            #     invalid_ms_count += 1
            continue
        else:
            extracted.append(normalized)

    valid_count = len(extracted)
    filtered_count = total_count - valid_count

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(extracted, f, ensure_ascii=False, indent=2)

    print(f"\n=== 处理结果 ===")
    print(f"输入记录数: {total_count}")
    print(f"有效记录数: {valid_count}")
    print(f"过滤记录数: {filtered_count}")
    print(f"  - 无效 SMILES: {invalid_smiles_count}")
    print(f"  - 无效 MS 数据: {invalid_ms_count}")
    print(f"输出文件: {output_path}")


if __name__ == "__main__":
    main()
