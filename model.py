import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ipdb
from opt_einsum import contract

def idx_to_onehot(target, num_class, confidence=None):
    sample_size, class_size = target.size(0), num_class
    if confidence is None:
        y = torch.zeros(sample_size, class_size,dtype=torch.float16).to(target)
        y = y.scatter_(1, torch.unsqueeze(target, dim=1), 1.)
    else:
        y = torch.ones(sample_size, class_size)
        y = y * (1 - confidence.data).unsqueeze(1).expand(-1, class_size)
        y[torch.arange(sample_size).long(), target.data] = confidence.data

    y = y.cuda()

    return y

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t, weight=None, soft_label_mask=None):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        if isinstance(y_t, list):
            p_t_list = [F.softmax(y / self.T, dim=1) for y in y_t]
            p_t_tensor = torch.stack(p_t_list)
            p_t = p_t_tensor.mean(0)
        else:
            p_t = F.softmax(y_t / self.T, dim=1)
        if weight is not None:
            loss = torch.sum(weight.unsqueeze(0).matmul(F.kl_div(p_s, p_t, reduction='none'))) * \
                   (self.T ** 2) / y_s.shape[0]
        elif soft_label_mask is not None:
            loss = torch.sum(F.kl_div(p_s*soft_label_mask, p_t*soft_label_mask, reduction='sum')) * \
                   (self.T ** 2) / y_s.shape[0]
        else:
            loss = torch.sum(F.kl_div(p_s, p_t, reduction='sum')) * \
                   (self.T ** 2) / y_s.shape[0]
        return loss

def cosinematrix(A):
    prod = torch.mm(A, A.t())  # 分子
    norm = torch.norm(A, dim=1).unsqueeze(0)  # 分母
    cos = prod.div(torch.mm(norm.t(), norm))
    return cos


def process_long_input(model, input_ids, attention_mask, start_tokens, end_tokens, token_type_ids=None):
    # Split the input to 2 overlapping chunks. Now BERT can encode inputs of which the length are up to 1024.
    n, c = input_ids.size()
    start_tokens = torch.tensor(start_tokens).to(input_ids)
    end_tokens = torch.tensor(end_tokens).to(input_ids)
    len_start = start_tokens.size(0)
    len_end = end_tokens.size(0)
    if c <= 512:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=True,
        )

        sequence_output = output[0]
        attention = output[-1][-1]
    else:
        new_input_ids, new_attention_mask, num_seg = [], [], []
        seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
        for i, l_i in enumerate(seq_len):
            if l_i <= 512:
                new_input_ids.append(input_ids[i, :512])
                new_attention_mask.append(attention_mask[i, :512])
                num_seg.append(1)
            else:
                input_ids1 = torch.cat([input_ids[i, :512 - len_end], end_tokens], dim=-1)
                input_ids2 = torch.cat([start_tokens, input_ids[i, (l_i - 512 + len_start): l_i]], dim=-1)
                attention_mask1 = attention_mask[i, :512]
                attention_mask2 = attention_mask[i, (l_i - 512): l_i]
                new_input_ids.extend([input_ids1, input_ids2])
                new_attention_mask.extend([attention_mask1, attention_mask2])
                num_seg.append(2)
        input_ids = torch.stack(new_input_ids, dim=0)
        attention_mask = torch.stack(new_attention_mask, dim=0)

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=True,
        )
        sequence_output = output[0]
        attention = output[-1][-1]
        i = 0
        new_output, new_attention = [], []
        for (n_s, l_i) in zip(num_seg, seq_len):
            if n_s == 1:
                output = F.pad(sequence_output[i], (0, 0, 0, c - 512))
                att = F.pad(attention[i], (0, c - 512, 0, c - 512))
                new_output.append(output)
                new_attention.append(att)
            elif n_s == 2:
                output1 = sequence_output[i][:512 - len_end]
                mask1 = attention_mask[i][:512 - len_end]
                att1 = attention[i][:, :512 - len_end, :512 - len_end]
                output1 = F.pad(output1, (0, 0, 0, c - 512 + len_end))
                mask1 = F.pad(mask1, (0, c - 512 + len_end))
                att1 = F.pad(att1, (0, c - 512 + len_end, 0, c - 512 + len_end))

                output2 = sequence_output[i + 1][len_start:]
                mask2 = attention_mask[i + 1][len_start:]
                att2 = attention[i + 1][:, len_start:, len_start:]
                output2 = F.pad(output2, (0, 0, l_i - 512 + len_start, c - l_i))
                mask2 = F.pad(mask2, (l_i - 512 + len_start, c - l_i))
                att2 = F.pad(att2, [l_i - 512 + len_start, c - l_i, l_i - 512 + len_start, c - l_i])
                mask = mask1 + mask2 + 1e-10
                output = (output1 + output2) / mask.unsqueeze(-1)
                att = (att1 + att2)
                att = att / (att.sum(-1, keepdim=True) + 1e-10)
                new_output.append(output)
                new_attention.append(att)
            i += n_s
        sequence_output = torch.stack(new_output, dim=0)
        attention = torch.stack(new_attention, dim=0)
    return sequence_output, attention


class Bert4SemiRE2(nn.Module):
    #结果意外的好
    def __init__(self, config, model, emb_size=768, fea_size=256, block_size=64, num_class=19):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.fea_size = fea_size

        self.head_extractor = nn.Linear(config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(config.hidden_size, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, fea_size)

        self.final_linear = nn.Linear(fea_size, num_class)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_class = num_class

        self.loss = F.nll_loss

    def encode(self, input_ids, attention_mask, subj_pos=None, obj_pos=None):

        head_extractor = self.head_extractor
        tail_extractor = self.tail_extractor
        bilinear = self.bilinear
        final_linear = self.final_linear
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        assert len(subj_pos) == len(obj_pos)
        hs, ts = [], []
        for i in range(len(subj_pos)):
            subj_pos_list = subj_pos[i] if isinstance(subj_pos[i], list) else [subj_pos[i]]
            obj_pos_list = obj_pos[i] if isinstance(obj_pos[i], list) else [obj_pos[i]]
            entity_head_list = sequence_output[i, subj_pos_list]
            entity_tail_list = sequence_output[i, obj_pos_list]
            hs.append(torch.mean(entity_head_list, dim=0).squeeze())
            ts.append(torch.mean(entity_tail_list, dim=0).squeeze())
        hs = torch.stack(hs, dim=0)
        ts = torch.stack(ts, dim=0)
        hs = torch.tanh(head_extractor(hs))
        ts = torch.tanh(tail_extractor(ts))
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        fea = torch.cat([bilinear(bl)], dim=1)
        logits = final_linear(fea)
        return logits, fea

    def forward(self,
                input_ids=None,
                attention_mask=None,
                subj_pos=None,
                obj_pos=None,
                relations=None,
                ):
        logits, fea = self.encode(input_ids, attention_mask, subj_pos=subj_pos, obj_pos=obj_pos)
        pred = F.log_softmax(logits, dim=1)
        if relations is not None:
            loss = self.loss(pred,relations)
        else:
            loss = torch.tensor(0)
        return logits, loss, fea

    def predict(self,
                input_ids=None,
                attention_mask=None,
                subj_pos=None,
                obj_pos=None,
                ):
        logits, fea = self.encode(input_ids, attention_mask, subj_pos=subj_pos, obj_pos=obj_pos)
        logits = logits.detach().cpu().numpy()
        predictions = np.argmax(logits, axis=1).tolist()
        confidences=[]
        for p in logits:
            confidences.append(max(p) ** 0.5)
        return predictions, confidences

class Bert4SemiRE3(nn.Module):
    #原模型
    def __init__(self, config, model, emb_size=768, fea_size=256, block_size=64, num_class=42):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.fea_size = fea_size

        self.head_extractor = nn.Linear(config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(config.hidden_size, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, fea_size)

        self.final_linear = nn.Linear(fea_size, num_class)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_class = num_class

        self.loss = F.nll_loss

    def encode(self, input_ids, attention_mask, subj_pos=None, obj_pos=None, is_mlm=True):
        head_extractor = self.head_extractor
        tail_extractor = self.tail_extractor
        bilinear = self.bilinear
        final_linear = self.final_linear
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        assert len(subj_pos) == len(obj_pos)
        hs, ts = [], []
        for i in range(len(subj_pos)):
            subj_pos_list = subj_pos[i] if isinstance(subj_pos[i], list) else [subj_pos[i]]
            obj_pos_list = obj_pos[i] if isinstance(obj_pos[i], list) else [obj_pos[i]]
            entity_head_list = sequence_output[i, subj_pos_list]
            entity_tail_list = sequence_output[i, obj_pos_list]
            hs.append(torch.mean(entity_head_list, dim=0).squeeze())
            ts.append(torch.mean(entity_tail_list, dim=0).squeeze())

        hs = torch.stack(hs, dim=0)
        ts = torch.stack(ts, dim=0)

        hs = torch.tanh(head_extractor(torch.cat([hs], dim=1)))
        ts = torch.tanh(tail_extractor(torch.cat([ts], dim=1)))
        # hs = torch.tanh(head_extractor(hs))
        # ts = torch.tanh(tail_extractor(ts))

        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        fea = torch.cat([bilinear(bl)], dim=1)
        logits = final_linear(fea)
        return logits, fea

    def forward(self,
                input_ids=None,
                attention_mask=None,
                subj_pos=None,
                obj_pos=None,
                relations=None,
                CE_loss_mask=None,
                ):
        logits, fea = self.encode(input_ids, attention_mask, subj_pos=subj_pos, obj_pos=obj_pos)
        pred = F.log_softmax(logits, dim=1)
        if relations is not None:
            if CE_loss_mask is None:
                loss = self.loss(pred,relations)
            else:
                select_id = (CE_loss_mask==1).nonzero().squeeze()
                loss = self.loss(pred[select_id], relations[select_id])
        else:
            loss = torch.tensor(0)
        return logits, loss

    def predict(self,
                input_ids=None,
                attention_mask=None,
                subj_pos=None,
                obj_pos=None,
                ):
        logits, fea = self.encode(input_ids, attention_mask, subj_pos=subj_pos, obj_pos=obj_pos)
        logits = logits.detach().cpu().numpy()
        predictions = np.argmax(logits, axis=1).tolist()
        confidences=[]
        for p in logits:
            confidences.append(max(p) ** 0.5)
        return predictions, confidences


class Bert4SemiRE3_adv(nn.Module):
    def __init__(self, config, model, emb_size=768, fea_size=256, block_size=64, num_class=19):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.fea_size = fea_size

        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, fea_size)

        self.final_linear = nn.Linear(fea_size, num_class)

        self.mlm_head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.mlm_tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.mlm_bilinear = nn.Linear(emb_size * block_size, fea_size)

        self.mlm_final_linear = nn.Linear(fea_size, num_class)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_class = num_class

        self.loss = F.nll_loss

    def encode(self, input_ids, attention_mask, subj_pos=None, obj_pos=None, is_mlm=False):
        if is_mlm:
            head_extractor = self.mlm_head_extractor
            tail_extractor = self.mlm_tail_extractor
            bilinear = self.mlm_bilinear
            final_linear = self.mlm_final_linear
        else:
            head_extractor = self.head_extractor
            tail_extractor = self.tail_extractor
            bilinear = self.bilinear
            final_linear = self.final_linear
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        assert len(subj_pos) == len(obj_pos)
        hs, ts, rs = [], [], []
        for i in range(len(subj_pos)):
            subj_pos_list = subj_pos[i] if isinstance(subj_pos[i], list) else [subj_pos[i]]
            obj_pos_list = obj_pos[i] if isinstance(obj_pos[i], list) else [obj_pos[i]]
            entity_head_list = sequence_output[i, subj_pos_list]
            entity_tail_list = sequence_output[i, obj_pos_list]
            hs.append(torch.mean(entity_head_list, dim=0).squeeze())
            ts.append(torch.mean(entity_tail_list, dim=0).squeeze())

            h_atts = torch.sum(attention[i, :, subj_pos_list], dim=1)
            t_atts = torch.sum(attention[i, :, obj_pos_list], dim=1)
            ht_att = (h_atts * t_atts).mean(0).squeeze()
            ht_att = ht_att / (ht_att.sum(0, keepdim=True) + 1e-5)
            rs.append(contract("rd,r->d", sequence_output[i], ht_att))

        hs = torch.stack(hs, dim=0)
        ts = torch.stack(ts, dim=0)
        rs = torch.stack(rs, dim=0)

        hs = torch.tanh(head_extractor(torch.cat([hs, rs], dim=1)))
        ts = torch.tanh(tail_extractor(torch.cat([ts, rs], dim=1)))
        # hs = torch.tanh(head_extractor(hs))
        # ts = torch.tanh(tail_extractor(ts))

        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        fea = torch.cat([bilinear(bl)], dim=1)
        logits = final_linear(fea)
        return logits, fea

    def forward(self,
                input_ids=None,
                attention_mask=None,
                subj_pos=None,
                obj_pos=None,
                relations=None,
                ):
        logits, fea = self.encode(input_ids, attention_mask, subj_pos=subj_pos, obj_pos=obj_pos)
        pred = F.log_softmax(logits, dim=1)
        if relations is not None:
            loss = self.loss(pred,relations)
        else:
            loss = torch.tensor(0)
        return logits, loss

    def predict(self,
                input_ids=None,
                attention_mask=None,
                subj_pos=None,
                obj_pos=None,
                ):
        logits, fea = self.encode(input_ids, attention_mask, subj_pos=subj_pos, obj_pos=obj_pos)
        logits = logits.detach().cpu().numpy()
        predictions = np.argmax(logits, axis=1).tolist()
        confidences=[]
        for p in logits:
            confidences.append(max(p) ** 0.5)
        return predictions, confidences

    def mlm(self,
                mlm_input_ids=None,
                mlm_attention_mask=None,
                mlm_subj_pos=None,
                mlm_obj_pos=None,
                relations=None,
                ):
        logits, fea = self.encode(mlm_input_ids, mlm_attention_mask, subj_pos=mlm_subj_pos, obj_pos=mlm_obj_pos, is_mlm=True)
        pred = F.log_softmax(logits, dim=1)
        if relations is not None:
            loss = self.loss(pred,relations)
        else:
            loss = torch.tensor(0)
        return logits, loss

class Bert4SemiRE_adv(nn.Module):
    def __init__(self, config, model, emb_size=768, fea_size=256, block_size=64, num_class=19):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.fea_size = fea_size

        self.head_extractor = nn.Linear(config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(config.hidden_size, emb_size)

        self.final_linear = nn.Linear(2 * emb_size, num_class)

        self.mlm_head_extractor = nn.Linear(config.hidden_size, emb_size)
        self.mlm_tail_extractor = nn.Linear(config.hidden_size, emb_size)

        self.mlm_final_linear = nn.Linear(2 * emb_size, num_class)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_class = num_class

        self.loss = F.nll_loss

    def encode(self, input_ids, attention_mask, subj_pos=None, obj_pos=None, is_mlm=False):
        if is_mlm:
            head_extractor = self.mlm_head_extractor
            tail_extractor = self.mlm_tail_extractor
            final_linear = self.mlm_final_linear
        else:
            head_extractor = self.head_extractor
            tail_extractor = self.tail_extractor
            final_linear = self.final_linear
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        assert len(subj_pos) == len(obj_pos)
        hs, ts = [], []
        for i in range(len(subj_pos)):
            subj_pos_list = subj_pos[i] if isinstance(subj_pos[i], list) else [subj_pos[i]]
            obj_pos_list = obj_pos[i] if isinstance(obj_pos[i], list) else [obj_pos[i]]
            entity_head_list = sequence_output[i, subj_pos_list]
            entity_tail_list = sequence_output[i, obj_pos_list]
            hs.append(torch.mean(entity_head_list, dim=0).squeeze())
            ts.append(torch.mean(entity_tail_list, dim=0).squeeze())
        hs = torch.stack(hs, dim=0)
        ts = torch.stack(ts, dim=0)
        hs = torch.tanh(head_extractor(hs))
        ts = torch.tanh(tail_extractor(ts))
        fea = torch.cat([hs, ts], dim=1)
        logits = final_linear(fea)
        return logits, fea

    def forward(self,
                input_ids=None,
                attention_mask=None,
                subj_pos=None,
                obj_pos=None,
                relations=None,
                ):
        logits, fea = self.encode(input_ids, attention_mask, subj_pos=subj_pos, obj_pos=obj_pos)
        pred = F.log_softmax(logits, dim=1)
        if relations is not None:
            loss = self.loss(pred, relations)
        else:
            loss = torch.tensor(0)
        return logits, loss

    def mlm(self,
            mlm_input_ids=None,
            mlm_attention_mask=None,
            mlm_subj_pos=None,
            mlm_obj_pos=None,
            relations=None,
            ):
        mlm_logits, mlm_fea = self.encode(mlm_input_ids, mlm_attention_mask, subj_pos=mlm_subj_pos,
                                          obj_pos=mlm_obj_pos, is_mlm=True)
        mlm_pred = F.log_softmax(mlm_logits, dim=1)
        if relations is not None:
            loss = self.loss(mlm_pred, relations)
        else:
            loss = torch.tensor(0)
        return mlm_logits, loss

    def predict(self,
                input_ids=None,
                attention_mask=None,
                subj_pos=None,
                obj_pos=None,
                ):
        logits, fea = self.encode(input_ids, attention_mask, subj_pos=subj_pos, obj_pos=obj_pos)
        logits = logits.detach().cpu()
        fea = fea.detach().cpu()
        predictions = np.argmax(logits.numpy(), axis=1).tolist()
        confidences = []
        for p in logits:
            confidences.append(max(p) ** 0.5)
        return predictions, confidences

class Bert4SemiRE_rl(nn.Module):
    def __init__(self, config, model, emb_size=768, fea_size=256, block_size=64, num_class=19):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.fea_size = fea_size

        self.final_linear = nn.Linear(2 * emb_size, num_class)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_class = num_class
        self.dropout = nn.Dropout()

        self.loss = F.nll_loss

    def encode(self, input_ids, attention_mask, subj_pos=None, obj_pos=None, is_mlm=False):
        head_extractor = self.head_extractor
        tail_extractor = self.tail_extractor
        final_linear = self.final_linear
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        assert len(subj_pos) == len(obj_pos)
        hs, ts = [], []
        for i in range(len(subj_pos)):
            subj_pos_list = subj_pos[i] if isinstance(subj_pos[i], list) else [subj_pos[i]]
            obj_pos_list = obj_pos[i] if isinstance(obj_pos[i], list) else [obj_pos[i]]
            entity_head_list = sequence_output[i, subj_pos_list]
            entity_tail_list = sequence_output[i, obj_pos_list]
            hs.append(torch.mean(entity_head_list, dim=0).squeeze())
            ts.append(torch.mean(entity_tail_list, dim=0).squeeze())
        hs = torch.stack(hs, dim=0)
        ts = torch.stack(ts, dim=0)
        hs = torch.tanh(head_extractor(hs))
        ts = torch.tanh(tail_extractor(ts))
        fea = torch.cat([hs, ts], dim=1)
        logits = final_linear(fea)
        return logits, fea

    def forward(self,
                input_ids=None,
                attention_mask=None,
                subj_pos=None,
                obj_pos=None,
                relations=None,
                ):
        logits, fea = self.encode(input_ids, attention_mask, subj_pos=subj_pos, obj_pos=obj_pos)
        pred = F.log_softmax(logits, dim=1)
        if relations is not None:
            loss = self.loss(pred, relations)
        else:
            loss = torch.tensor(0)
        return logits, loss

    def mlm(self,
            mlm_input_ids=None,
            mlm_attention_mask=None,
            mlm_subj_pos=None,
            mlm_obj_pos=None,
            relations=None,
            ):
        mlm_logits, mlm_fea = self.encode(mlm_input_ids, mlm_attention_mask, subj_pos=mlm_subj_pos,
                                          obj_pos=mlm_obj_pos, is_mlm=True)
        mlm_pred = F.log_softmax(mlm_logits, dim=1)
        if relations is not None:
            loss = self.loss(mlm_pred, relations)
        else:
            loss = torch.tensor(0)
        return mlm_logits, loss

    def predict(self,
                input_ids=None,
                attention_mask=None,
                subj_pos=None,
                obj_pos=None,
                ):
        logits, fea = self.encode(input_ids, attention_mask, subj_pos=subj_pos, obj_pos=obj_pos)
        logits = logits.detach().cpu()
        fea = fea.detach().cpu()
        predictions = np.argmax(logits.numpy(), axis=1).tolist()
        confidences = []
        for p in logits:
            confidences.append(max(p) ** 0.5)
        return predictions, confidences




class Bert4SemiRE_mlm(nn.Module):
    def __init__(self, config, model, emb_size=768, fea_size=256, block_size=64, num_class=19):
        super().__init__()
        self.config = config
        self.bert_mlm = model

    def forward(self,
                input_ids=None,
                attention_mask=None,
                ):
        logits = self.bert_mlm(input_ids,attention_mask=attention_mask)[0]
        mask_id = (input_ids==103).nonzero(as_tuple=False)[:,1].tolist()
        pred_ids = torch.max(logits,dim=2).indices
        input_ids[0,mask_id] = pred_ids[0,mask_id]
        return input_ids

