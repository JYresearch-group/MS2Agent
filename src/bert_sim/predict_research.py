# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/22 17:57
# software: PyCharm

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import json
from utils import ClassificationDataPreprocess, compute_metrics, trans_data, trans_data_embedding
from argparse import Namespace
from trainer import Trainer
import torch
from tqdm import tqdm


def bijiao(line, t):

    if line['smiles'] == t['smiles']:
        return True

    if line['spectrumid'] == t['candidate_id']:
        return True

    return False


def read_jsonl(file):

    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            x1 = f"analysis_type: ms_retrieval_signature\nrecord_id: {line['res_ms_ana']['analysis_row']['record_id']}\n"
            # x2 = f"analysis_type: structure_retrieval_signature\nrecord_id: {line['record_id']}\n"
            line['ms_summary_text'] = line['res_ms_ana']['analysis_row']['ms_summary_text'].replace(x1, "")
            # line['structure_summary_text'] = line['structure_summary_text'].replace(x2, "")

            res111 = []
            true_label_id = []
            for i, t in enumerate(line['retrieval_top50_res_smile_ana']):
                x2 = f"analysis_type: structure_retrieval_signature\nrecord_id: {t['res_smile_ana']['analysis_row']['record_id']}\n"
                res111.append({"structure_summary_text": t['res_smile_ana']['analysis_row']['structure_summary_text'].replace(x2, "")})

                if bijiao(line, t):  # line['smiles'] == t['smiles']:
                    true_label_id.append(i)

            line['retrieval_top50_structure_summary_text'] = res111
            line['true_label_id'] = true_label_id
            data.append(line)

    return data


class LanguageModelClassificationPredict(ClassificationDataPreprocess):

    def __init__(self, config_file_name):
        config = json.load(open(os.path.join(config_file_name, 'classification_config.json'), 'r', encoding='utf-8'))
        self.config = Namespace(**config)
        super(LanguageModelClassificationPredict, self).__init__(self.config)
        self.label_id = self.config.label_id
        self.label_0 = self.config.labels[0]
        self.model_flag = self.config.model_flag
        self.max_seq_len = self.config.max_seq_len
        self.trainer = Trainer(self.config)
        self.trainer.load_model()

    def process(self, text, texts=None):

        if self.model_flag == "embedding":

            res_tokens = []
            res_token_type_ids = []
            res_attention_mask = []
            res_labels = []

            for t in text['retrieval_top50_structure_summary_text']:
                d = {
                    "text_a": text['ms_summary_text'],
                    "text_b": t['structure_summary_text'],
                    "label": self.label_0,
                    "model_flag": self.model_flag
                }
                text_a = d['text_a']
                text_b = d['text_b']
                embedding_a = self.get_embedding([text_a])
                embedding_b = self.get_embedding([text_b])
                d['embedding_a'] = embedding_a
                d['embedding_b'] = embedding_b

                t1 = trans_data_embedding(d['embedding_a'], d['embedding_b'])
                res_tokens.append(t1)
                res_labels.append([1, 0])

            res_tokens = torch.cat(res_tokens, dim=0)
            # res_token_type_ids = torch.tensor(res_token_type_ids)
            # res_attention_mask = torch.tensor(res_attention_mask)
            res_token_type_ids = res_tokens
            res_attention_mask = res_tokens
            res_labels = torch.tensor(res_labels)

        else:
            res_tokens = []
            res_token_type_ids = []
            res_attention_mask = []
            res_labels = []

            for t in text['retrieval_top50_structure_summary_text']:
                d = {
                    "text_a": text['ms_summary_text'],
                    "text_b": t['structure_summary_text'],
                    "label": self.label_0,
                    "model_flag": self.model_flag
                }
                d1 = self.text_to_ids(d)
                d.update(d1)
                # data.append(d)
                t1, t3 = trans_data(d['input_ids_a'], d['input_ids_b'], max_seq_len1=self.max_seq_len)
                t2, t3 = trans_data(d['token_type_ids_a'], d['token_type_ids_b'], max_seq_len1=self.max_seq_len)

                res_tokens.append(t1)
                res_token_type_ids.append(t2)
                res_attention_mask.append(t3)
                res_labels.append([1, 0])

            res_tokens = torch.tensor(res_tokens)
            res_token_type_ids = torch.tensor(res_token_type_ids)
            res_attention_mask = torch.tensor(res_attention_mask)
            res_labels = torch.tensor(res_labels)

        res = {"input_ids": res_tokens, "token_type_ids": res_token_type_ids, "attention_mask": res_attention_mask,
               "label": res_labels}

        return res

    def select_topk(self, data, ids, topk=(1, 3, 5, 10, 20)):
        res1 = {"top1": 0, "top3": 0, "top5": 0, "top10": 0, "top20": 0, "top100": 0, "top200": 0, "top300": 0, "top400": 0, "top500": 0}
        data1 = [[i, d] for i, d in enumerate(data)]
        data2 = sorted(data1, key=lambda x: x[1], reverse=True)
        data3 = [d[0] for d in data2]

        for i in ids:
            print("i ...", ids, data3.index(i), data2[data3.index(i)])
            if i in data3[:1]:
                res1['top1'] = 1
            if i in data3[:3]:
                res1['top3'] = 1
            if i in data3[:5]:
                res1['top5'] = 1
            if i in data3[:10]:
                res1['top10'] = 1
            if i in data3[:20]:
                res1['top20'] = 1
            if i in data3[:100]:
                res1['top100'] = 1
            if i in data3[:200]:
                res1['top200'] = 1
            if i in data3[:300]:
                res1['top300'] = 1
            if i in data3[:400]:
                res1['top400'] = 1
            if i in data3[:500]:
                res1['top500'] = 1
        return res1, data3

    def predict(self, texts, pred_file=None):

        res1 = {"top1": 0, "top3": 0, "top5": 0, "top10": 0, "top20": 0, "top100": 0, "top200": 0, "top300": 0, "top400": 0, "top500": 0}

        res111 = []
        for i, text in tqdm(enumerate(texts), desc='process ...'):
            test_data = self.process(text)
            intent_preds_list, intent_preds_list_pr, intent_preds_list_all = self.trainer.evaluate_test(test_data)
            res_temp_all = [x['1'] for x in intent_preds_list_all]  # {'1': 0.9973390698432922, '0': 0.002624473301693797}
            r, ttt11 = self.select_topk(res_temp_all, text['true_label_id'])

            for k, v in res1.items():
                res1[k] += r[k]

            res111.append({
                "id": i,
                "index": ttt11
            })

        res2 = {k: v/len(texts) for k, v in res1.items()}

        print("res2 ...", res2)

        if pred_file:
            folder_path = os.path.dirname(pred_file)

            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)

            out_file = os.path.join(folder_path, f"res_index_{len(res111)}.json")

            f = open(out_file, 'w', encoding='utf-8')
            json.dump(res111, f, ensure_ascii=False, indent=4)
            f.close()

        return res2


if __name__ == '__main__':

    file = "./output/model_bert_1"  # 修改模型地址
    lcp = LanguageModelClassificationPredict(file)

    pred_file = ["/data0/xj/Slime_MS/llm_ms_slime_v3/outputs/casmi_retrieval_signature_research_gnps_qwen3.5-122b-a10b_v5/res_research_analyses.jsonl",
                 '/data0/xj/Slime_MS/llm_ms_slime_v3/outputs/casmi_retrieval_signature_research_gnps_11_qwen3.5-122b-a10b_v6/res_research_analyses.jsonl',
                 '/data0/xj/Slime_MS/llm_ms_slime_v3/outputs/casmi_retrieval_signature_research_gnps_17_qwen3.5-122b-a10b_v6/res_research_analyses.jsonl'][-1]

    pred_file = ["/Users/xiaojie/Documents/lunwen/Slime_MS/llm_ms_slime_v3/outputs/casmi_retrieval_signature_research_gnps_qwen3.5-122b-a10b_v5/res_research_analyses.jsonl",
                 '/Users/xiaojie/Documents/lunwen/Slime_MS/llm_ms_slime_v3/outputs/casmi_retrieval_signature_research_gnps_11_qwen3.5-122b-a10b_v6/res_research_analyses.jsonl',
                 '/Users/xiaojie/Documents/lunwen/Slime_MS/llm_ms_slime_v3/outputs/casmi_retrieval_signature_research_gnps_17_qwen3.5-122b-a10b_v6/res_research_analyses.jsonl',
                 '/Users/xiaojie/Documents/lunwen/Slime_MS/llm_ms_slime_v3/outputs/casmi_retrieval_signature_research_gnps_C15H22O3_10_200_openai_gpt-5.4-mini-2026-03-17_v6/res_research_analyses.jsonl',
                 '/root/llm_ms_slime_v3/outputs/casmi_retrieval_signature_siyou_data_20_qwen3.5-4b_v6/res_research_analyses.jsonl',
                 '/root/llm_ms_slime_v3/outputs/casmi_retrieval_signature_siyou_data_20_zijian_76_qwen3.5-4b_v6/res_research_analyses.jsonl'][-2]

    data = read_jsonl(pred_file)
    res = lcp.predict(data, pred_file)

