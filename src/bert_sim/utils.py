# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/22 17:57
# software: PyCharm


from transformers import BertTokenizer, BertConfig, AlbertConfig, AlbertTokenizer, RobertaConfig, RobertaTokenizer, \
    XLNetConfig, XLNetTokenizer, XLNetModel
from transformers import BertModel, BertPreTrainedModel, RobertaModel, AlbertModel
from config import MODEL_CLASSES
import logging
import copy
import json
import numpy as np
import torch
import random
from tqdm import tqdm
from torch.utils.data import TensorDataset, Dataset, DataLoader
from mertics import metrics_report
from mertics1 import txt_classification_metric
from typing import Tuple
# from text_augment import word_level_augment
from embedding_utils import load_sentence_transformer


logger = logging.getLogger(__name__)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        label: (Optional) string. The intent label of the example.
    """

    def __init__(self, guid, text, text_b=None, label=None, label1=None):
        self.guid = guid
        self.text = text
        self.text_b = text_b
        self.label = label
        self.label1 = label1

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, labels=None, labels1=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.labels = labels
        self.labels1 = labels1

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


class ClassificationDataPreprocess:

    def __init__(self, config=None):

        self.config = config
        self.ADDITIONAL_SPECIAL_TOKENS = self.config.ADDITIONAL_SPECIAL_TOKENS
        self.ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]

        if self.config.model_flag == 'bert':
            self.tokenizer = self.load_tokenizer(self.config)

        if self.config.model_type == 'embedding':
            self.embedding_model = load_sentence_transformer(self.config.embedding_model_path)

    def load_tokenizer(self, args):
        if args.model_type in ["albert", "roberta", 'albert_tiny']:
            tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
            return tokenizer
        tokenizer = MODEL_CLASSES[args.model_type][1].from_pretrained(args.model_name_or_path)
        tokenizer.add_special_tokens({"additional_special_tokens": self.ADDITIONAL_SPECIAL_TOKENS})
        return tokenizer
    
    def _get_data_file(self, file, model_flag, set_type='train'):
        # with open(file, 'r', encoding='utf-8') as f:
        #     lines = f.readlines()
        lines = json.load(open(file, 'r', encoding='utf-8'))
        data = []
        labels = []
        for line in lines:
            # line = json.loads(line)
            data.append({
                "text_a": line['ms_summary_text'],
                "text_b": line['structure_summary_text'],
                "label": '1',
                "model_flag": model_flag,
                "set_type": set_type
            })
            # labels += line['label']
        # labels = sorted(list(set(labels)))
        labels = ['1', '0']
        
        return data, labels
    
    # def _get_data_file(self, file):
    #
    #     with open(file, 'r', encoding='utf-8') as f:
    #         lines = f.readlines()
    #
    #     res = []
    #     labels = []
    #     text_len = []
    #     for line in lines:
    #         line = json.loads(line)
    #         # text = line["text"]
    #         # aspect = line['aspect']['word']
    #
    #         text = line['word']["word"]
    #         aspect = line["aspect_type"]
    #
    #         if len(text) + len(aspect) > self.config.max_seq_len - 10:
    #
    #             inxs1 = text.index(aspect) if aspect in text else 0
    #
    #             nn0 = (self.config.max_seq_len - 10 - len(aspect))
    #             nn1 = int(nn0/2)
    #             a = max(inxs1 - nn1, 0)
    #             b = inxs1 + len(aspect) + nn1
    #
    #             text = text[a: b]
    #
    #         l1 = line['label']
    #
    #         if l1 not in labels:
    #             labels.append(l1)
    #
    #         if len(text) < 1:
    #             continue
    #
    #         text_len.append(len(text))
    #         res.append({
    #             "text": text,
    #             "text1": aspect,
    #             "label": l1,
    #             "label1": l1
    #         })
    #
    #     print(f"max len: {max(text_len)}, ... min len: {min(text_len)} ..., mean len : {sum(text_len) / len(text_len)} ... the length : {len(text_len)} ...")
    #     return res, labels
    
    def convert_examples_to_features(self, examples, max_seq_len, tokenizer,
                                     cls_token='[CLS]',
                                     sep_token='[SEP]',
                                     pad_token=0,
                                     pad_token_label_id=-100,
                                     cls_token_segment_id=0,
                                     pad_token_segment_id=0,
                                     sequence_a_segment_id=0,
                                     sequence_b_segment_id=1,
                                     mask_padding_with_zero=True):
        # Setting based on the current model type
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        # unk_token = tokenizer.unk_token
        pad_token_id = tokenizer.pad_token_id

        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 5000 == 0:
                # logger.info("Writing example %d of %d" % (ex_index, len(examples)))
                print("Writing example %d of %d" % (ex_index, len(examples)))

            # Tokenize word by word

            tokens = tokenizer.tokenize(example.text)
            token_type_ids = [sequence_a_segment_id] * len(tokens)

            # tokens_b = tokenizer.tokenize(example.text_b)
            # token_type_ids_b = [sequence_b_segment_id] * len(tokens_b)

            tokens = [cls_token] + tokens
            token_type_ids = [cls_token_segment_id] + token_type_ids

            # Add [SEP] token
            tokens += [sep_token]
            token_type_ids += [sequence_a_segment_id]

            # Add B
            # tokens += tokens_b
            # token_type_ids += token_type_ids_b

            # Add [SEP] token
            # tokens += [sep_token]
            # token_type_ids += [sequence_b_segment_id]

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_len - len(input_ids)

            if padding_length > 0:

                input_ids = input_ids + ([pad_token_id] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            elif padding_length < 0:
                input_ids = input_ids[:max_seq_len-1] + [input_ids[-1]]
                attention_mask = attention_mask[:max_seq_len-1] + [attention_mask[-1]]
                token_type_ids = token_type_ids[:max_seq_len-1] + [token_type_ids[-1]]

            assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
            assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
                len(attention_mask), max_seq_len)
            assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(
                len(token_type_ids), max_seq_len)

            label_id = example.label

            if ex_index < 5:
                print("example")
                logger.info("*** Example ***")
                logger.info("guid: %s" % example.guid)
                logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                logger.info("intent_label: %s " % " ".join([str(x) for x in label_id]))

            features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              labels=label_id,
                              labels1=label_id
                              ))

        return features

    def text_to_ids(self, data, sequence_a_segment_id=0, sequence_b_segment_id=1, cls_token_segment_id=0):

        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        # unk_token = tokenizer.unk_token
        pad_token_id = self.tokenizer.pad_token_id

        text_a = data['text_a']
        text_b = data['text_b']

        tokens_a = self.tokenizer.tokenize(text_a)
        token_type_ids_a = [sequence_a_segment_id] * len(tokens_a)

        tokens_a = [cls_token] + tokens_a + [sep_token]
        token_type_ids_a = [cls_token_segment_id] + token_type_ids_a + [sequence_a_segment_id]
        input_ids_a = self.tokenizer.convert_tokens_to_ids(tokens_a)

        tokens_b = self.tokenizer.tokenize(text_b)
        token_type_ids_b = [sequence_b_segment_id] * len(tokens_b)

        # Add [SEP] token
        tokens_b += [sep_token]
        token_type_ids_b += [sequence_b_segment_id]
        input_ids_b = self.tokenizer.convert_tokens_to_ids(tokens_b)

        return {'input_ids_a': input_ids_a, 'input_ids_b': input_ids_b,
                'token_type_ids_a': token_type_ids_a, 'token_type_ids_b': token_type_ids_b}


    def collate_fnction(self, ):

        print(1)

    def _get_data(self, data, label_id, set_type="train"):

        res = SimData(data, self.config)
        # res1 = Dataset(res, c)
        return res

    def get_embedding(self, ms_texts, batch_size=16):

        show_progress_bar = False
        if len(list(ms_texts)) > batch_size*10:
            show_progress_bar = True

        ms_embeddings = self.embedding_model.encode(
            list(ms_texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
        )

        return ms_embeddings


def trans_data(res1, res2, max_seq_len1=512):

    data1 = res1[:-1]
    data2 = res2[:-1]
    max_seq_len = max_seq_len1 - 2

    if len(data1) + len(data2) > max_seq_len:

        m = len(data2) + len(data1) - max_seq_len

        while len(data1) + len(data2) > max_seq_len:
            if len(data1) > len(data2):
                data1.pop()
            else:
                data2.pop()

    data1 = data1 + [res1[-1]]
    data2 = data2 + [res2[-1]]

    t1 = data1 + data2
    t2 = [1]*len(t1)
    t1 = t1 + [0]*(max_seq_len1 - len(t1))
    t2 = t2 + [0]*(max_seq_len1 - len(t2))
    return t1, t2


def trans_data_embedding(res1, res2):
    res1 = torch.tensor(res1)
    res2 = torch.tensor(res2)
    res = torch.cat([res1, res2], dim=1)

    return res


def collect_fuction(data, max_seq_len=512, top_k=3):
    max_seq_len = data[0]['max_seq_len']

    if data[0]['model_flag'] == 'bert':
        if data[0]['set_type'] == 'train' or data[0]['set_type'] == 'dev':
            top_k = min(top_k, len(data))
            res_tokens = []
            res_token_type_ids = []
            res_attention_mask = []
            res_labels = []

            for i, d in enumerate(data):
                t1, t3 = trans_data(d['input_ids_a'], d['input_ids_b'], max_seq_len1=max_seq_len)
                t2, t3 = trans_data(d['token_type_ids_a'], d['token_type_ids_b'], max_seq_len1=max_seq_len)

                res_tokens.append(t1)
                res_token_type_ids.append(t2)
                res_attention_mask.append(t3)
                res_labels.append([1, 0])

                ddd = data[:i] + data[i+1:]
                data1 = random.sample(ddd, min(top_k, len(ddd)))
                for d1 in data1:
                    t1, t3 = trans_data(d['input_ids_a'], d1['input_ids_b'], max_seq_len1=max_seq_len)
                    t2, t3 = trans_data(d['token_type_ids_a'], d1['token_type_ids_b'], max_seq_len1=max_seq_len)

                    res_tokens.append(t1)
                    res_token_type_ids.append(t2)
                    res_attention_mask.append(t3)
                    res_labels.append([0, 1])

            res_tokens = torch.tensor(res_tokens)
            res_token_type_ids = torch.tensor(res_token_type_ids)
            res_attention_mask = torch.tensor(res_attention_mask)
            res_labels = torch.tensor(res_labels)

            res = {"input_ids": res_tokens, "token_type_ids": res_token_type_ids, "attention_mask": res_attention_mask, "label": res_labels}

            return res

    elif data[0]['model_flag'] == 'embedding':
        if data[0]['set_type'] == 'train' or data[0]['set_type'] == 'dev':
            top_k = min(top_k, len(data))
            res_tokens = []
            res_token_type_ids = []
            res_attention_mask = []
            res_labels = []

            for i, d in enumerate(data):
                t1 = trans_data_embedding(d['embedding_a'], d['embedding_b'])

                res_tokens.append(t1)
                # res_token_type_ids.append(t2)
                # res_attention_mask.append(t3)
                res_labels.append([1, 0])

                ddd = data[:i] + data[i + 1:]
                data1 = random.sample(ddd, min(top_k, len(ddd)))
                for d1 in data1:
                    t1 = trans_data_embedding(d['embedding_a'], d1['embedding_b'])
                    res_tokens.append(t1)
                    # res_token_type_ids.append(t2)
                    # res_attention_mask.append(t3)
                    res_labels.append([0, 1])

            res_tokens = torch.cat(res_tokens, dim=0)
            # res_token_type_ids = torch.tensor(res_token_type_ids)
            # res_attention_mask = torch.tensor(res_attention_mask)
            res_labels = torch.tensor(res_labels)

            res = {"input_ids": res_tokens, "token_type_ids": res_tokens, "attention_mask": res_tokens,
                   "label": res_labels}

            return res


class SimData(Dataset):
    def __init__(self, data, config):
        self.data = data
        self.process = ClassificationDataPreprocess(config)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index]
        d['max_seq_len'] = self.process.config.max_seq_len
        if d['model_flag'] == 'bert':
            d1 = self.process.text_to_ids(d)
            d.update(d1)

        elif d['model_flag'] == "embedding":
            text_a = d['text_a']
            text_b = d['text_b']
            embedding_a = self.process.get_embedding([text_a])
            embedding_b = self.process.get_embedding([text_b])
            d['embedding_a'] = embedding_a
            d['embedding_b'] = embedding_b
        return d


def compute_metrics(intent_preds_list, out_intent_label_list):
    # metrics, report = metrics_report(out_intent_label_list, intent_preds_list)
    texts = ["a"]*len(intent_preds_list)
    rpt_info, cf_mat_df, std_pred_cmp_infos = txt_classification_metric(intent_preds_list, out_intent_label_list, texts)

    # y_true = out_intent_label_list
    # y_pred = intent_preds_list
    # from sklearn.metrics import accuracy_score
    # print("*"*80)
    # acc1 = accuracy_score(y_true, y_pred)
    # print("acc1 ..", acc1)
    #
    # from sklearn.metrics import balanced_accuracy_score
    # acc2 = balanced_accuracy_score(y_true, y_pred)
    # print("acc2 ..", acc2)
    #
    # from sklearn.metrics import classification_report
    #
    # target_names = sorted(list(set(y_true)))
    # print("sklearn ....")
    # print(classification_report(y_true, y_pred, target_names=target_names))
    metrics = copy.deepcopy(rpt_info)
    report = rpt_info

    metrics['mean'] = {
        "mean_precision": metrics['total']['precision'],
        "mean_recall": metrics['total']['recall'],
        "mean_f1-score": metrics['total']['f1'],
    }

    metrics['sum'] = {
        "sum_precision": metrics['total']['precision'],
        "sum_recall": metrics['total']['recall'],
        "sum_f1-score": metrics['total']['f1'],
    }

    return metrics, report


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by 2
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param

