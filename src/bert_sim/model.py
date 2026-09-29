# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/22 18:11
# software: PyCharm


from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn
from config import MODEL_CLASSES
import torch
import torch.nn.functional as F
import numpy as np
import random
# from info_nce import InfoNCE, info_nce


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.5, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()
        # self.softmax = nn.Softmax(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.training:
            x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        x1 = self.linear(x)
        # x1 = self.softmax(x1)
        x1 = self.sigmoid(x1)
        return x1


class BertPool(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class SelfAttention(nn.Module):

    def __init__(self, sentence_num=0, key_size=0, hidden_size=0, attn_dropout=0.1):

        super(SelfAttention, self).__init__()
        self.linear_k = nn.Linear(hidden_size, key_size, bias=False)
        self.linear_q = nn.Linear(hidden_size, key_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dim_k = np.power(key_size, 0.5)
        self.softmax = nn.Softmax(1)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x, mask=None):
        """
        :param x:  [batch_size, max_seq_len, embedding_size]
        :param mask:
        :return:   [batch_size, embedding_size]
        """
        k = self.linear_k(x)
        q = self.linear_q(x)
        v = self.linear_v(x)
        # f = self.softmax(q.matmul(k.t()) / self.dim_k)
        attn = torch.bmm(q, k.transpose(1, 2)) / self.dim_k
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class FCLayer1(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0., use_activation=True):
        super(FCLayer1, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        if self.training:
            x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class LSTMEncoder(nn.Module):
    def __init__(self, sent_rep_size, sent_hidden_size, sent_num_layers, dropout, bidirectional=True):
        '''
        LSTM编码器，用于对句子的编码(含MASK)
        输入：[batch_size, sequence_len, sent_rep_size]
        输出：[batch_size, sequence_len, sent_hidden_size*2]
        :param sent_rep_size: int, 输入句子的embedding size
        :param sent_hidden_size: int, LSTM的隐藏层的size
        :param sent_num_layers: int, LSTM的层数
        :param dropout: float，dropout的比例
        '''
        super(LSTMEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.sent_lstm = nn.LSTM(
            input_size=sent_rep_size,
            hidden_size=sent_hidden_size,
            num_layers=sent_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        )

    def forward(self, sent_reps, sent_masks):
        # sent_reps:  b x doc_len x sent_rep_size
        # sent_masks: b x doc_len

        sent_hiddens, _ = self.sent_lstm(sent_reps)  # b x doc_len x hidden*2
        sent_hiddens = sent_hiddens * sent_masks.unsqueeze(2)

        if self.training:
            sent_hiddens = self.dropout(sent_hiddens)

        return sent_hiddens


class GRUEncoder(nn.Module):
    def __init__(self, sent_rep_size, sent_hidden_size, sent_num_layers, dropout, bidirectional=True):
        '''
        LSTM编码器，用于对句子的编码(含MASK)
        输入：[batch_size, sequence_len, sent_rep_size]
        输出：[batch_size, sequence_len, sent_hidden_size*2]
        :param sent_rep_size: int, 输入句子的embedding size
        :param sent_hidden_size: int, LSTM的隐藏层的size
        :param sent_num_layers: int, LSTM的层数
        :param dropout: float，dropout的比例
        '''
        super(GRUEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.sent_lstm = nn.GRU(
            input_size=sent_rep_size,
            hidden_size=sent_hidden_size,
            num_layers=sent_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        )

    def forward(self, sent_reps, sent_masks):
        # sent_reps:  b x doc_len x sent_rep_size
        # sent_masks: b x doc_len

        sent_hiddens, _ = self.sent_lstm(sent_reps)  # b x doc_len x hidden*2
        sent_hiddens = sent_hiddens * sent_masks.unsqueeze(2)

        if self.training:
            sent_hiddens = self.dropout(sent_hiddens)

        return sent_hiddens


class TextCNNEncoder(nn.Module):
    def __init__(self, word_dims, out_channel, filter_sizes, dropout):
        super(TextCNNEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.word_dims = word_dims

        # self.word_embed = nn.Embedding(vocab.word_size, self.word_dims, padding_idx=0)

        # extword_embed = vocab.load_pretrained_embs(word2vec_path)
        # extword_size, word_dims = extword_embed.shape
        # logging.info("Load extword embed: words %d, dims %d." % (extword_size, word_dims))
        #
        # self.extword_embed = nn.Embedding(extword_size, word_dims, padding_idx=0)
        # self.extword_embed.weight.data.copy_(torch.from_numpy(extword_embed))
        # self.extword_embed.weight.requires_grad = False

        input_size = self.word_dims

        self.filter_sizes = filter_sizes # [2, 3, 4]  # n-gram window
        self.out_channel = out_channel  # 100
        self.convs = nn.ModuleList([nn.Conv2d(1, self.out_channel, (filter_size, input_size), bias=True)
                                    for filter_size in self.filter_sizes])

    def forward(self, batch_embed):
        # word_ids: sen_num x sent_len
        # extword_ids: sen_num x sent_len
        # batch_masks: sen_num x sent_len
        sen_num, sent_len, embedding_size = batch_embed.shape
        # print("sen_num:{}".format(sen_num))
        # print("sent_len:{}".format(sent_len))
        # print("embedding_size:{}".format(embedding_size))

        # word_embed = self.word_embed(word_ids)  # sen_num x sent_len x 100
        # extword_embed = self.extword_embed(extword_ids)
        # batch_embed = word_embed + extword_embed

        if self.training:
            batch_embed = self.dropout(batch_embed)

        batch_embed.unsqueeze_(1)  # sen_num x 1 x sent_len x 100

        pooled_outputs = []
        for i in range(len(self.filter_sizes)):
            filter_height = sent_len - self.filter_sizes[i] + 1
            conv = self.convs[i](batch_embed)
            hidden = F.relu(conv)  # sen_num x out_channel x filter_height x 1

            mp = nn.MaxPool2d((filter_height, 1))  # (filter_height, filter_width)
            pooled = mp(hidden).reshape(sen_num, self.out_channel)  # sen_num x out_channel x 1 x 1 -> sen_num x out_channel

            pooled_outputs.append(pooled)

        reps = torch.cat(pooled_outputs, dim=1)  # sen_num x total_out_channel

        if self.training:
            reps = self.dropout(reps)

        return reps


class ClassificationModel(BertPreTrainedModel):

    @staticmethod
    def repair_bert_embedding_buffers(model):
        """
        After `from_pretrained` on this class (Transformers 5.x meta init + nested BERT load),
        `bert.embeddings.position_ids` / `token_type_ids` non-persistent buffers can be left invalid.
        Re-register them like `BertEmbeddings.__init__` does.
        """
        if not hasattr(model, "bert") or not hasattr(model.bert, "embeddings"):
            return
        emb = model.bert.embeddings
        cfg = model.bert.config
        dev = emb.word_embeddings.weight.device
        n = cfg.max_position_embeddings
        position_ids = torch.arange(n, device=dev, dtype=torch.long).expand((1, -1))
        token_type_ids = torch.zeros((1, n), dtype=torch.long, device=dev)
        emb.register_buffer("position_ids", position_ids, persistent=False)
        emb.register_buffer("token_type_ids", token_type_ids, persistent=False)

    def __init__(self, model_dir, args):
        self.args = args
        self.label_num = args.num_classes

        self.config_class, _, config_model = MODEL_CLASSES[args.model_type]
        bert_config = self.config_class.from_pretrained(args.model_name_or_path)

        self.bert_config_output_hidden_states = False
        # bert_config.output_attentions = True
        # bert_config.output_hidden_states = True
        if bert_config.output_hidden_states:
            self.bert_config_output_hidden_states = True

        super(ClassificationModel, self).__init__(bert_config)

        # Transformers >=5 wraps model __init__ in `torch.device("meta")` during from_pretrained.
        # Nested `BertModel.from_pretrained` must run on a real device (e.g. CPU), otherwise it raises.
        with torch.device("cpu"):
            self.bert = config_model.from_pretrained(args.model_name_or_path, config=bert_config)  # Load pretrained bert
        
        # for param in self.bert.parameters():
        #     param.requires_grad = True

        # self.pooling = BertPool(bert_config)

        # self.rnn = LSTMEncoder(bert_config.hidden_size, 128, 1, self.args.dropout_rate)
        # self.rnn = nn.LSTM(self.args.embedding_dim,
        #                    self.args.hidden_dim,
        #                    num_layers=self.args.n_layers,
        #                    bidirectional=self.args.bidirectional,
        #                    dropout=self.args.dropout_rate)

        # out_channel = 100
        # filter_sizes = [2, 3, 4]
        # self.cnn = TextCNNEncoder(bert_config.hidden_size, out_channel, filter_sizes, self.args.dropout_rate)

        # attention
        # self.att = SelfAttention(sentence_num=34, key_size=bert_config.hidden_size, hidden_size=bert_config.hidden_size)

        # if args.is_lstm:
        #     self.rnn = LSTMEncoder(bert_config.hidden_size, self.args.hidden_dim, self.args.n_layers,
        #                            self.args.dropout_rate, bidirectional=args.bidirectional)
        # 
        # elif args.is_gru:
        #     self.rnn = GRUEncoder(bert_config.hidden_size, self.args.hidden_dim, self.args.n_layers, self.args.dropout_rate)

        # self.fc = FCLayer(128*2, self.label_num)

        # if args.is_lstm:
        #     if args.bidirectional:
        #         self.fc = FCLayer(self.args.hidden_dim * 2, self.args.num_classes, self.args.dropout_rate)
        #     else:
        #         self.fc = FCLayer(self.args.hidden_dim, self.args.num_classes, self.args.dropout_rate)
        # 
        # elif args.is_gru:
        #     self.fc = FCLayer(self.args.hidden_dim * 2, self.args.num_classes, self.args.dropout_rate)
        # 
        # else:
        #     self.fc = FCLayer(bert_config.hidden_size, self.label_num)

        self.fc = FCLayer(bert_config.hidden_size, self.label_num)
        
        # self.fc2 = FCLayer(bert_config.hidden_size * 2, self.label_num)
        # # self.fc1 = FCLayer(bert_config.hidden_size, self.label_num)
        # # self.fc2 = FCLayer(bert_config.hidden_size, self.label_num)
        # # self.fc3 = FCLayer(bert_config.hidden_size, self.label_num)
        # self.fc3 = FCLayer1(self.args.max_seq_len, 1)
        
        # self.fc = FCLayer(300, self.label_num, self.args.dropout_rate)
        # if self.label_num == 35:
        #     self.fc1 = FCLayer(300, 8, self.args.dropout_rate)
        #
        #     self.fc2 = FCLayer(300*2, self.label_num, self.args.dropout_rate)
        #
        #     self.fc3 = FCLayer1(bert_config.hidden_size, 300, self.args.dropout_rate)
        #     self.fc4 = FCLayer1(300, 300, self.args.dropout_rate)
        #
        # if self.label_num == 8:
        #     self.fc = FCLayer(300, 8, self.args.dropout_rate)
        
        # self.fc_8_mlp = FCLayer1(300*2, 300, self.args.dropout_rate)
        # self.fc_35_mlp = FCLayer1(300*2, 300, self.args.dropout_rate)
        # 
        # self.fc_8_mlp1 = FCLayer1(300, 300*2, self.args.dropout_rate)
        # self.fc_35_mlp1 = FCLayer1(300, 300 * 2, self.args.dropout_rate)

        # loss
        self.loss_fct_cros = nn.CrossEntropyLoss()
        self.loss_fct_bce = nn.BCELoss()
        self.loss_fct_bce1 = nn.BCELoss(reduction='none')
        # self.nce_loss = InfoNCE(negative_mode='unpaired')

        # Required by Transformers 5.x (sets all_tied_weights_keys, ties weights; skips full re-init under meta context)
        self.post_init()

    def info_nce_loss(self, out, label):

        label_cus = {}
        label_index = torch.argmax(label, dim=1)

        for i in range(label.shape[0]):
            l = label_index[i]
            l = l.cpu().numpy().tolist()
            if l not in label_cus:
                label_cus[l] = []
            label_cus[l].append(i)

        if len(label_cus) == 1:
            return False
        
        output = None
        n = 0
        for k, v in label_cus.items():
            if len(v) > 1:
                random.shuffle(v)
                all_keys = []
                for i in v:
                    all_keys.append(out[i, :].unsqueeze(dim=0))
                all_keys = torch.cat(all_keys, dim=0)
                
                a = len(v) // 2
                query = all_keys[:a, :]
                positive_key = all_keys[a:a+a, :]

                # query = out[v[0], :].unsqueeze(dim=0)
                # positive_key = []
                # for x in v[1:]:
                #     positive_key.append(out[x, :].unsqueeze(dim=0))
                # positive_key = torch.cat(positive_key, dim=0)
                
                negative_keys = []
                for i in range(label.shape[0]):
                    if i not in v:
                        negative_keys.append(out[i, :].unsqueeze(dim=0))
                        
                negative_keys = torch.cat(negative_keys, dim=0)
                if output is None:
                    output = self.nce_loss(query, positive_key, negative_keys)
                else:
                    output += self.nce_loss(query, positive_key, negative_keys)
                n += 1
        output = output/n
        return output

    def forward(self, input_ids, attention_mask, token_type_ids, label=None, label1=None):
        batch_size = input_ids.shape[0]
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        sequence_output = outputs[0]  # [batch_size, max_sen_len, embedding_size]
        out = outputs[1]  # [CLS]  [batch_size, embedding_size]

        # if self.args.is_lstm or self.args.is_gru:
        #     hiddens = self.rnn(sequence_output, attention_mask)
        #
        #     # output = self.cnn(input_embed)  # (16, 768)
        #     out = torch.mean(hiddens, dim=1)

        # out = outputs[0]
        # if self.label_num == 35:
        #     out1 = self.fc3(outputs[1])
        #
        #     # out = self.rnn(sequence_output, attention_mask)
        #     #
        #     # out = torch.mean(out, dim=1)
        #
        #     out = self.cnn(out)
        #
        #     out2 = self.fc4(out)
        #
        #     out = torch.cat([out1, out2], dim=1)
        #     # out = out1 + out2
        #
        #     # out_8 = self.fc_8_mlp(out)
        #     # out_8 = self.fc_8_mlp1(out_8)
        #     #
        #     # out_35 = self.fc_35_mlp(out)
        #     # out_35 = self.fc_35_mlp1(out_35)
        #     #
        #     # logits1 = self.fc1(out_8)
        #
        #     logits0 = self.fc2(out)
        #
        # if self.label_num == 8:
        #     out = self.cnn(out)
        #     logits0 = self.fc(out)

        # logits0 = F.log_softmax(logits0, dim=-1)
        logits0 = self.fc(out)
        logits = logits0

        if label is not None:
            label = label.to(logits.dtype)
            if self.label_num == 1:
                logits = logits.squeeze(-1)
                loss = self.loss_fct_bce(logits, label)
            else:
                # loss = self.loss_fct_cros(logits.view(-1, self.label_num), label.view(-1))
                loss = self.loss_fct_bce(logits, label)

            # label1 = label1.unsqueeze(dim=-1)
            # label1 = torch.zeros(batch_size, 8).to(label1.device).scatter_(1, label1, 1)
            # loss1 = self.loss_fct_bce(logits1, label1)

            # loss1 = self.loss_fct_cros(logits1, label1)
            # nce_loss = self.info_nce_loss(out, label)
            # if nce_loss:
            #     loss = loss + nce_loss
            #     print("nce loss", nce_loss)

            # loss = loss + loss1
            outputs = (loss,) + (logits,)

        else:
            outputs = logits

        return outputs  # (loss), logits


class SimMLPModel(nn.Module):
    def __init__(self, args):
        super(SimMLPModel, self).__init__()
        self.label_num = args.num_classes
        self.args = args
        self.fc1 = FCLayer(args.embedding_hidden_size * 2, args.embedding_hidden_size)
        # self.fc2 = nn.Linear(args.embedding_hidden_size, args.embedding_hidden_size_1, bias=False)
        self.fc3 = FCLayer(args.embedding_hidden_size, self.label_num)

        self.loss_fct_cros = nn.CrossEntropyLoss()
        self.loss_fct_bce = nn.BCELoss()
        self.loss_fct_bce1 = nn.BCELoss(reduction='none')

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, label=None, label1=None):

        out = input_ids
        out1 = self.fc1(out)
        # out2 = self.fc2(out1)
        logits0 = self.fc3(out1)

        logits = logits0

        if label is not None:
            label = label.to(logits.dtype)
            if self.label_num == 1:
                logits = logits.squeeze(-1)
                loss = self.loss_fct_bce(logits, label)
            else:
                # loss = self.loss_fct_cros(logits.view(-1, self.label_num), label.view(-1))
                loss = self.loss_fct_bce(logits, label)

            # label1 = label1.unsqueeze(dim=-1)
            # label1 = torch.zeros(batch_size, 8).to(label1.device).scatter_(1, label1, 1)
            # loss1 = self.loss_fct_bce(logits1, label1)

            # loss1 = self.loss_fct_cros(logits1, label1)
            # nce_loss = self.info_nce_loss(out, label)
            # if nce_loss:
            #     loss = loss + nce_loss
            #     print("nce loss", nce_loss)

            # loss = loss + loss1
            outputs = (loss,) + (logits,)

        else:
            outputs = logits

        return outputs  # (loss), logits

