# -*- conding: utf-8 -*-
# @Time    : 2026/4/5  18:01
# @Author  : psi


import json


def read_file():
    data1 = json.load(open("./data/CASMI2016_Cat2and3_Challenge_1-neg.instrument_fields_smiles.json", 'r', encoding='utf-8'))
    data2 = json.load(open("./data/CASMI2016_Cat2and3_Challenge_1-pos.instrument_fields_smiles.json", 'r', encoding='utf-8'))

    data = data1 + data2

    f = open("./data/CASMI2016_Cat2and3_Challenge_1-all.instrument_fields_smiles.json", 'w', encoding='utf-8')
    json.dump(data, f, ensure_ascii=False, indent=2)
    f.close()
    print(f"the save success is : {len(data)} ...")


if __name__ == '__main__':
    read_file()


