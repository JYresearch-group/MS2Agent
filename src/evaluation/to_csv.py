# -*- conding: utf-8 -*-
# @Time    : 2026/5/14  22:09
# @Author  : psi

import json
import copy
import pandas as pd


def read_file(file):
    data = json.load(open(file, 'r', encoding='utf-8'))

    return data


def read_jsonl(file):

    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    return data


if __name__ == '__main__':

    # file0 = "/Users/xiaojie/Documents/lunwen/Slime_MS/data/MS2-SAN/GNPS-NIH-NATURALPRODUCTSLIBRARY_ROUND2_POSITIVE_new_C15H22O3_10.json"

    # file1 = "/Users/xiaojie/Documents/lunwen/Slime_MS/llm_ms_slime/outputs/C15H22O3_10_blind_pipeline_ms2_carbonyl_balanced_rerank.json"

    """
    === outputs/C15H22O3_10_blind_pipeline_ms2_carbonyl_balanced_rerank.json ===
    hit_index0: [195, 90, 63, 40, 8, 104, 5, 59, 22, 158]
    hit_rank1 : [196, 91, 64, 41, 9, 105, 6, 60, 23, 159]
    hits      : 10/10
    rank      : min=6, median=62.0, mean=75.40, max=196
    recall@1  : 0/10 = 0.0000
    recall@3  : 0/10 = 0.0000
    recall@5  : 0/10 = 0.0000
    recall@10 : 2/10 = 0.2000
    recall@50 : 4/10 = 0.4000
    recall@199: 10/10 = 1.0000
    """

    file2 = ["/Users/xiaojie/Documents/lunwen/Slime_MS/llm_ms_slime_v3/data/GNPS-NIH-NATURALPRODUCTSLIBRARY_ROUND2_POSITIVE_new_C15H22O3_10_200_new.json",
             '/root/llm_ms_slime_v3/data/siyou_data_20_formula_pubchem_candidates_new_200.json',
             '/root/llm_ms_slime_v3/data/siyou_data_20_candidates_new_76.json'][-2]

    file3 = ["/Users/xiaojie/Documents/lunwen/Slime_MS/llm_ms_slime_v3/outputs/casmi_retrieval_signature_research_gnps_C15H22O3_10_200_openai_gpt-5.4-mini-2026-03-17_v6/res_index_10.json",
             "/root/llm_ms_slime_v3/outputs/casmi_retrieval_signature_siyou_data_20_qwen3.5-4b_v6/res_index_20.json"][-1]

    file4 = ["/Users/xiaojie/Documents/lunwen/Slime_MS/llm_ms_slime_v3/outputs/casmi_retrieval_signature_research_gnps_C15H22O3_10_200_openai_gpt-5.4-mini-2026-03-17_v6/retrieval_results_new_bert_100.json",
             "/root/llm_ms_slime_v3/outputs/casmi_retrieval_signature_siyou_data_20_qwen3.5-4b_v6/retrieval_results_new_bert_40.json"][-1]

    file5 = ["/Users/xiaojie/Documents/lunwen/Slime_MS/llm_ms_slime_v3/outputs/casmi_retrieval_signature_research_gnps_C15H22O3_10_200_openai_gpt-5.4-mini-2026-03-17_v6/res_research_analyses.jsonl",
             "/root/llm_ms_slime_v3/outputs/casmi_retrieval_signature_siyou_data_20_qwen3.5-4b_v6/res_research_analyses.jsonl"][-1]

    # data0 = read_file(file0)
    # data1 = read_file(file1)
    data2 = read_file(file2)
    data3 = read_file(file3)
    data4 = read_file(file4)
    data5 = read_jsonl(file5)

    res = []
    for d in data2:
        dd00 = {
            "ms": json.dumps(d['mz_lists']),
            'pepmass': d['pepmass'],
            'charge': d['charge'],
            'mslevel': d['mslevel'],
            'source_instrument': d['source_instrument'],
            'filename': d['name'],
            'seq': d.get('seq', ""),
            'ionmode': d['ionmode'],
            'organism': d.get('organism', ""),
            'name': d['name'],
            'pi': d.get('pi', ""),
            'datacollector': d.get('datacollector', ""),
            'smiles': d['smiles'],
            'inchi': d.get('inchi', ""),
            'inchiaux': d.get('inchiaux', ""),
            'pubmed': d.get('pubmed', ""),
            'submituser': d.get('submituser', ""),
            'libraryquality': d.get('libraryquality', ""),
            'spectrumid': d['spectrumid'],
            'usi': d.get('usi', ""),
            'scans': d['scans'],
            'spectrumid_old': d.get('spectrumid_old', ""),
            'adduct': d['adduct'],
        }
        for i, r in enumerate(d['retrieval_top50']):
            r1 = {"res1_"+k: json.dumps(v) for k, v in r.items()}

            xx = copy.deepcopy(dd00)
            xx.update(r1)

            xx['res1_id'] = i

            res.append(xx)

    res3 = []
    for d in data3:
        for i, x in enumerate(d['index']):
            res3.append({
                "sample_id": d['id'],
                "id": i,
                "index": x
            })

    res4 = []
    for d in data4['per_query_rankings']:
        for i, x in enumerate(d['raw_ranked_record_ids']):
            res4.append({
                'target_record_id': d['target_record_id'],
                "id": i,
                "index": x
            })

    res5 = []
    res51 = {}
    for d in data4['llm_rerank']['embedding_only']['per_query_results']:
        for i, x in enumerate(d['ranking_result']['ranked_record_ids']):
            res5.append({
                'target_record_id': d['target_record_id'],
                "id": i,
                "index": x
            })
            if d['target_record_id'] not in res51:
                res51[d['target_record_id']] = []
            res51[d['target_record_id']].append({
                'target_record_id': d['target_record_id'],
                "id": i,
                "index": x
            })

    df1 = pd.DataFrame(res)
    df2 = pd.DataFrame(res3)
    df3 = pd.DataFrame(res4)
    df4 = pd.DataFrame(res5)

    out_file = ["/Users/xiaojie/Documents/lunwen/Slime_MS/llm_ms_slime_v3/outputs/casmi_retrieval_signature_research_gnps_C15H22O3_10_200_openai_gpt-5.4-mini-2026-03-17_v6/",
                "/root/llm_ms_slime_v3/outputs/casmi_retrieval_signature_siyou_data_20_qwen3.5-4b_v6/old/"][-1]

    out_file1 = out_file + "res_1.csv"
    out_file2 = out_file + "res_2.csv"
    out_file3 = out_file + "res_3.csv"
    out_file4 = out_file + "res_4.csv"
    
    df1.to_csv(out_file1, index=False)
    df2.to_csv(out_file2, index=False)
    df3.to_csv(out_file3, index=False)
    df4.to_csv(out_file4, index=False)

    res6 = []
    for d in data5:
        # if d['spectrumid'] in ['t23815386', 't24123446']:
        dd00 = {
            "ms": json.dumps(d['mz_lists']),
            'pepmass': d['pepmass'],
            'charge': d['charge'],
            'mslevel': d['mslevel'],
            'source_instrument': d['source_instrument'],
            'filename': d['name'],
            'seq': d.get('seq', ""),
            'ionmode': d['ionmode'],
            'organism': d.get('organism', ""),
            'name': d['name'],
            'pi': d.get('pi', ""),
            'datacollector': d.get('datacollector', ""),
            'smiles': d['smiles'],
            'inchi': d.get('inchi', ""),
            'inchiaux': d.get('inchiaux', ""),
            'pubmed': d.get('pubmed', ""),
            'submituser': d.get('submituser', ""),
            'libraryquality': d.get('libraryquality', ""),
            'spectrumid': d['spectrumid'],
            'usi': d.get('usi', ""),
            'scans': d['scans'],
            'spectrumid_old': d.get('spectrumid_old', ""),
            'adduct': d['adduct'],
            'ms_summary_text': d['res_ms_ana']['analysis_row']['ms_summary_text']
        }
        for i, x in enumerate(d['retrieval_top50_res_smile_ana']):
            xx = copy.deepcopy(dd00)
            x1 = {"res1_" + k: v for k, v in x.items() if k != 'res_smile_ana' or k != 'ms2_formula_filter'}
            x1['res1_id'] = i
            x1['res1_structure_summary_text'] = x['res_smile_ana']['analysis_row']['structure_summary_text']
            xx.update(x1)
            res6.append(xx)

    df5 = pd.DataFrame(res6)
    out_file5 = out_file + "res_5.csv"
    df5.to_csv(out_file5, index=False)


    print(1)

