# -*- conding: utf-8 -*-
# @Time    : 2026/5/5  17:57
# @Author  : psi


import json


def read_file(file):
    if file.endswith('.jsonl'):
        data = []
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    else:
        data = json.load(open(file, 'r', encoding='utf-8'))

    return data


def save_json(data, file):
    json.dump(data, open(file, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    print(f"save success the file is: {file} ")


def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"the save success is {path} ...")



file1 = ['./data/GNPS-NIH-NATURALPRODUCTSLIBRARY_ROUND2_POSITIVE_new_C15H20O3_11_pepmass_MH_top930_ppm5_guaranteed_candidate_id_spectrumid.json_1.json',
         './data/GNPS-NIH-NATURALPRODUCTSLIBRARY_ROUND2_POSITIVE_new_with_adduct_50_pepmass_MH_calibrated_guaranteed_candidate_id_spectrumid.json_17.json'][0]

file2 = ["/Users/xiaojie/Documents/lunwen/Slime_MS/llm_ms_slime_v3/outputs/casmi_retrieval_signature_research_gnps_11_qwen3.5-122b-a10b_v6/res_research_analyses.jsonl",
         "/Users/xiaojie/Documents/lunwen/Slime_MS/llm_ms_slime_v3/outputs/casmi_retrieval_signature_research_gnps_17_qwen3.5-122b-a10b_v6/res_research_analyses.jsonl"][0]

file3 = ["../llm_ms_slime/outputs/C15H20O3_11_pepmass_MH_ppm5_msfilter_branch35_top215.json",
         "../llm_ms_slime/outputs/with_adduct_50_pepmass_MH_ppm5_msfilter_branch35_top599.json"][0]

file4 = "../llm_ms_slime/outputs/C15H20O3_11_pepmass_MH_ppm5_msfilter_branch35_grouped_top203.json"

data1 = read_file(file1)
data3 = read_file(file3)

data2 = read_file(file2)

data4 = read_file(file4)
# data3 = data4
res1 = []
res2 = []
for x1, x2, x3 in zip(data1, data2, data3['results']):

    assert x1['spectrumid'][1:] == x3['record_id'], 'error'
    assert x1['smiles'] == x3['target']['smiles'], 'error'
    smiles = x1['smiles']
    retrieval_top50 = x1['retrieval_top50']
    candidates = x3['candidates']

    assert smiles in [x['smiles'] for x in candidates], 'error1'

    t3_database_id = [x['database_id'] for x in candidates]
    t1_database_id = [x['database_id'] for x in retrieval_top50]

    assert len(set(t3_database_id).difference(set(t1_database_id))) == 0, 'error2'

    t1_in_t3_id = [i for i, x in enumerate(retrieval_top50) if x['database_id'] in t3_database_id]

    t_retrieval_top50 = [retrieval_top50[i] for i in t1_in_t3_id]
    x1['retrieval_top50'] = t_retrieval_top50
    res1.append(x1)

    retrieval_top50_2 = x2['retrieval_top50']
    t_retrieval_top50_2 = [retrieval_top50_2[i] for i in t1_in_t3_id]
    x2['retrieval_top50'] = t_retrieval_top50_2

    retrieval_top50_res_smile_ana = x2['retrieval_top50_res_smile_ana']
    t_retrieval_top50_res_smile_ana = [retrieval_top50_res_smile_ana[i] for i in t1_in_t3_id]
    x2['retrieval_top50_res_smile_ana'] = t_retrieval_top50_res_smile_ana
    res2.append(x2)

    assert [c['database_id'] for c in t_retrieval_top50] == [c['database_id'] for c in t_retrieval_top50_2], 'error3'
    assert [c['database_id'] for c in t_retrieval_top50] == [c['database_id'] for c in t_retrieval_top50_res_smile_ana], 'error4'


save_json(res1, file1)
save_jsonl(res2, file2)


print(1)
