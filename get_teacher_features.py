import argparse
import os

import torch
from utils import set_seed, collate_fn
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel, BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import numpy as np
from utils import score,print_table
import math
from apex import amp
from prepro import read_semeval, read_tacred
from model import Bert4SemiRE2
# from DualRE_model import Bert4SemiRE_predictor2
# from stanfordcorenlp import StanfordCoreNLP
import ipdb
import pickle as pkl
import re


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# nlp = StanfordCoreNLP(r'D:\stanford_nlp\stanford-corenlp-4.0.0')
entity_adj,predicate,prep,entity_type = {},{},{},{}
VB_list=['VB','VBN','VBD','VBG','VBP','VBZ']#VBN:-ed,VBD:is
IN_list=['IN','RB','POS']
no_words = ['that']
punctuation = '!,;:?."'
id_list = []

def cosinematrix(fea):
    fea_T = fea.transpose()
    adj = []
    model_num = fea.shape[1]
    for i in range(model_num):
        prod = np.matmul(fea[:,i,:],fea_T[:,i,:])
        norm = np.expand_dims(np.linalg.norm(fea[:,i,:], axis=1),axis=1)  # 检查标准化是否正确
        adj.append(prod/(np.matmul(norm,norm.transpose())))
    return adj

def evaluate(args, model, features, tag="dev", use_pos=False):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds, golds = [], []
    for batch in dataloader:
        model.eval()
        golds.extend(batch[4])
        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'subj_pos': batch[2],
                  'obj_pos': batch[3],
                  'relations': torch.tensor(batch[4], dtype=torch.long).to(args.device),
                  # 'pos_attention': batch[6] if use_pos else None,
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = np.argmax(preds, 1).tolist()
    prec_micro, recall_micro, f1_micro = score(golds, preds, tag=tag)

    return prec_micro, recall_micro, f1_micro

def get_entity(sentence):

    t1 = r'<e1>(.+)</e1>'
    t2 = r'<e2>(.+)</e2>'

    pattern1 = re.compile(t1)
    pattern2 = re.compile(t2)

    match1 = pattern1.search(sentence)
    match2 = pattern2.search(sentence)

    entity1 = match1.group(1)
    entity2 = match2.group(1)
    entity1 = entity1.split()[-1]
    entity2 = entity2.split()[-1]

    # return id,entity1,entity2,en1_pos_start,en1_pos_end,en2_pos_start,en2_pos_end,sentence
    return entity1,entity2

def calc_ent(x):
    """
        calculate shanno ent of x
    """
    x = x-min(x)+1e-10
    p = x/sum(x)
    logp = np.log2(p)
    ent = sum(-p * logp)
    return ent

def get_weight_list(logits_list, use_confidence=False, is_default=True):
    if is_default:
        return [1., 1.]
    elif use_confidence:
        confidences = [max(logit) ** 0.5 for logit in logits_list]
        return confidences/sum(confidences)
    else:
        confidences = [calc_ent(logit) for logit in logits_list]
        return confidences / sum(confidences)

def get_teacher_data(args, t_model_list, train_features, infer_features, dev_features, test_features):
    def pre_data(features):
        fea_list = [[]for i in range(len(t_model_list))]
        dataloader = DataLoader(features, batch_size=1, shuffle=False, collate_fn=collate_fn,
                                      drop_last=False)
        for step, batch in enumerate(dataloader):
            id_list.extend(batch[5])
            temp_logits_list = []
            for model_num, t_model in enumerate(t_model_list):
                t_model.eval()
                # input_ids, input_mask, subj_pos, obj_pos, relations, id, train_len
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'subj_pos': batch[2],
                          'obj_pos': batch[3],
                          'relations': torch.tensor(batch[4], dtype=torch.long).to(args.device),
                          # 'pos_attention': batch[6] if model_num == args.pos_model_num else None,
                          }

                logits, loss, fea = t_model(**inputs)
                fea_list[model_num].append(fea.detach().cpu().numpy())
                soft_logits = logits.squeeze().detach().cpu().numpy()
                temp_logits_list.append(soft_logits)
                features[step]['teacher_logits_{}'.format(model_num)] = soft_logits
            logits_weight = get_weight_list(temp_logits_list)
            features[step]['logits_weight'] = logits_weight
        return features, fea_list

    def get_case(features):
        dataloader = DataLoader(features, batch_size=1, shuffle=False, collate_fn=collate_fn,
                                drop_last=False)
        for step, batch in enumerate(dataloader):
            id_list.extend(batch[5])
            temp_logits_list = []
            for model_num, t_model in enumerate(t_model_list):
                t_model.eval()
                # input_ids, input_mask, subj_pos, obj_pos, relations, id, train_len
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'subj_pos': batch[2],
                          'obj_pos': batch[3],
                          'relations': torch.tensor(batch[4], dtype=torch.long).to(args.device),
                          # 'pos_attention': batch[6] if model_num == args.pos_model_num else None,
                          }

                logits, loss, fea = t_model(**inputs)
                soft_logits = logits.squeeze().detach().cpu().numpy()
                temp_logits_list.append(soft_logits)
                features[step]['teacher_logits_{}'.format(model_num)] = soft_logits
        return features

    train_features, train_fea_list = pre_data(train_features)
    infer_features, infer_fea_list = pre_data(infer_features)

    # test_features = get_case(test_features)
    # train_fea = np.concatenate(train_fea_list, 1)
    # infer_fea = np.concatenate(infer_fea_list, 1)
    # final_fea = np.concatenate([train_fea, infer_fea],0)
    return train_features, infer_features, dev_features, test_features, None

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="../Bert4SemiRE/dataset/semeval/", type=str)
    parser.add_argument("--feature_dir", default="./features_{}.pkl", type=str)

    parser.add_argument("--model_name_or_path", default="../Bert4SemiRE/bert/BERT-BASE-UNCASED",type=str)
    # parser.add_argument("--model_name_or_path", default="D:/Document-level relation extraction/DSDocRE-main/bert/BERT-BASE-UNCASED",type=str)
    # parser.add_argument("--model_name_or_path", default="../DSDocRE-main/bert/BERT-BASE-UNCASED",type=str)

    parser.add_argument("--train_batch_size", default=20, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=20, type=int,
                        help="Batch size for testing.")

    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--num_class", type=int, default=19)
    parser.add_argument("--model_num", type=int, default=2)
    parser.add_argument("--pos_model_num", type=int, default=1)
    parser.add_argument("--labeled_ratio", type=float, default=0.1)
    parser.add_argument("--unlabeled_ratio", type=float, default=0.5)

    parser.add_argument("--teacher_path", default="./saved_models/{}/Bert4SemiRE_model_teacher_{}.pt", type=str)

    parser.add_argument("--seed", type=int, default=2,
                        help="random seed for initialization")
    parser.add_argument('--use_entropy', action='store_true', default=False)
    parser.add_argument('--cuda', type=bool, default=True)

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_class,)
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path,)

    read = read_semeval

    train_file = os.path.join(args.data_dir+'train-' + str(args.labeled_ratio) + '.json')
    infer_file = os.path.join(args.data_dir+'raw-' + str(args.unlabeled_ratio) + '.json')
    dev_file = os.path.join(args.data_dir+'dev.json')
    test_file = os.path.join(args.data_dir+'test.json')

    train_features = read(train_file, tokenizer)
    infer_features = read(infer_file, tokenizer)


    dev_features = read(dev_file, tokenizer)
    test_features = read(test_file, tokenizer)

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)

    t_model_list = []
    # with open(args.feature_dir,'rb') as f:
    #     train_features, infer_features, dev_features, test_features = pkl.load(f)

    for i in range(args.model_num):
        model = BertModel.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
        # if i ==1:
        #     t_model = Bert4SemiRE(config, model, fea_size=300)
        #     t_model.to(0)
        # else:
        #     t_model = Bert4SemiRE(config, model)
        #     t_model.to(0)
        t_model = Bert4SemiRE2(config, model, num_class=args.num_class)
        t_model.to(0)

        t_model = amp.initialize(t_model, opt_level="O1", verbosity=0)
        t_model.load_state_dict(torch.load(args.teacher_path.format(args.labeled_ratio,i)))
        # t_model.load_state_dict(torch.load(args.teacher_path.format(i)))
        # evaluate(args, t_model, test_features, tag="test")
        t_model_list.append(t_model)

    train_features, infer_features, dev_features, test_features, final_fea = get_teacher_data(args, t_model_list, train_features,
                                                                                   infer_features, dev_features,
                                                                                   test_features)
    # adj = cosinematrix(final_fea)
    print('train_features length:',len(train_features))
    print('infer_features length:',len(infer_features))
    with open(args.feature_dir.format(args.labeled_ratio),'wb') as f:
        pkl.dump((train_features, infer_features, dev_features, test_features, None, id_list), f)

if __name__ == "__main__":
    main()