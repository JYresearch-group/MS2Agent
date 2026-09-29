# -*- conding: utf-8 -*-
# @Time    : 2026/5/4  22:25
# @Author  : psi


import json


def read_file(file):

    data = json.load(open(file, 'r', encoding='utf-8'))
    res_num = 0

    res = []
    res_1122 = []
    for d in data:
        smile = d['smiles']
        spectrumid = d['spectrumid']

        ids11 = ""
        res11 = []
        res22 = []
        for c in d['retrieval_top50']:
            c['candidate_id'] = "t"+c['database_id']
            if smile == c['smiles']:
                ids11 = c['candidate_id']
            res11.append(c)

            res22.append(abs(c['exact_mass']-d['pepmass']))

        print("max(res22) ...", max(res22))
        if ids11 != "":
            res_num += 1

        d['spectrumid'] = ids11
        d['retrieval_top50'] = res11

        res_1122.append(len(res11))

        if len(res11) > 200:
            continue

        res.append(d)
    res_1122 = sorted(res_1122)
    print(res_num)
    assert res_num == len(data), 'error'

    f = open(file+f"_{len(res)}.json", 'w', encoding='utf-8')
    json.dump(res, f, ensure_ascii=False, indent=2)
    f.close()


# read_file("./data/GNPS-NIH-NATURALPRODUCTSLIBRARY_ROUND2_POSITIVE_new_C15H20O3_11_pepmass_MH_top930_ppm5_guaranteed_candidate_id_spectrumid.json")

read_file("./data/GNPS-NIH-NATURALPRODUCTSLIBRARY_ROUND2_POSITIVE_new_with_adduct_50_pepmass_MH_calibrated_guaranteed_candidate_id_spectrumid.json")
