# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/22 17:56
# software: PyCharm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# os.environ["WORLD_SIZE"] = "1"

from utils import ClassificationDataPreprocess, init_logger
from argparse import Namespace
from trainer import Trainer
import logging
import os
import json
import codecs

init_logger()
logger = logging.getLogger(__name__)


class LanguageModelClassificationTrain(ClassificationDataPreprocess):

    def __init__(self, config_params):
        self.config = Namespace(**config_params)
        self.config.no_cuda = False
        self.model_save_path = self.config.model_save_path

        super(LanguageModelClassificationTrain, self).__init__(self.config)

    def data_preprocess(self):

        train_data, labels1 = self._get_data_file(self.config.train_file_url, self.config.model_flag, "train")

        # test_data, labels2 = self._get_data_file(self.config.test_file_url)
        dev_data, labels3 = self._get_data_file(self.config.dev_file_url, self.config.model_flag, "dev")
        labels2 = labels3
        test_data = dev_data

        # train_data = train_data[:100]
        # dev_data = dev_data[:50]
        # test_data = test_data[:50]

        labels = ["1", "0"]

        self.labels = labels
        self.config.num_classes = len(labels)

        logger.info("self.config.num_classes: {} ".format(str(self.config.num_classes)))

        self.label_id = {l: ind for ind, l in enumerate(labels)}
        self.id_label = {ind: l for ind, l in enumerate(labels)}
        self.config.label_id = self.label_id
        self.config.id_label = self.id_label
        self.config.labels = self.labels

        self.train_data = self._get_data(train_data, self.label_id, set_type="train")

        logger.info("train data num: {} ".format(str(len(train_data))))
        self.test_data = self._get_data(test_data, self.label_id, set_type="test")
        logger.info("test data num: {} ".format(str(len(test_data))))
        self.dev_data = self._get_data(dev_data, self.label_id, set_type="dev")
        logger.info("dev data num: {} ".format(str(len(dev_data))))

    def fit(self):

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path, exist_ok=True)
        self.config.model_save_path = self.model_save_path
        self.config.model_dir = self.model_save_path

        with codecs.open(os.path.join(self.model_save_path, '{}_config.json'.format(self.config.task_type)), 'w', encoding='utf-8') as fd:
            json.dump(vars(self.config), fd, indent=4, ensure_ascii=False)

        self.trainer = Trainer(self.config,
                               train_dataset=self.train_data,
                               dev_dataset=self.dev_data,
                               test_dataset=self.test_data)
        self.trainer.train()

    def eval(self):
        self.trainer.load_model()
        test_results = self.trainer.evaluate("test")
        return test_results


if __name__ == '__main__':
    config_params = {
        "algorithm_id": 19,
        "hyper_param_strategy": "CUSTOMED",
        "ADDITIONAL_SPECIAL_TOKENS": [],
        "model_dir": "./output",
        "data_dir": "./data",
        "model_type": "bert",
        "task_type": "classification",
        "seed": 1234,
        "train_batch_size": 16,
        "eval_batch_size": 16,
        "max_seq_len": 512,
        "learning_rate": 5e-5,
        "num_train_epochs": 3,
        "weight_decay": 0.0,
        "gradient_accumulation_steps": 1,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "max_steps": -1,
        "warmup_steps": 0,
        "dropout_rate": 0.1,
        "logging_steps": 200,
        "save_steps": 200,

        "is_gru": False,
        "is_lstm": False,
        "hidden_dim": 128,
        "n_layers": 1,
        "bidirectional": True,

        "no_cuda": False,
        "ignore_index": 0,
        "do_train": True,
        "do_eval": True,
        "train_file_url": "./o_data/train_12290.json",
        "dev_file_url": "./o_data/dev_3073.json",
        "test_file_url": "./o_data/dev_3073.json",
        "model_save_path": "./output/model_2",
        "model_flag": ["bert", "embedding"][-1],
        "embedding_model_path": ["/data3/xj/pre_model/Qwen3-Embedding-8B", "/data3/xj/pre_model/Qwen3-Embedding-0.6B"][-1],
        "embedding_hidden_size": [4096, 1024][-1],
        "embedding_hidden_size_1": [1024, 256][-1],
    }

    config_params['model_type'] = ["bert", "roberta", "albert_tiny", "embedding"][-1]  # 选择模型
    config_params['model_name_or_path'] = ["/data0/xj/pred_model/bert-base-uncased"][-1]  # 与训练模型

    config_params['model_save_path'] = ["./output/model_bert_1", "./output/model_embedding_0.6B_1"][-1]  # 修改模型保存路径
    lc = LanguageModelClassificationTrain(config_params)
    lc.data_preprocess()
    lc.fit()

    # nohup python3 -u train.py > log_train_model_embedding_0.6B_1.log 2>&1 &
