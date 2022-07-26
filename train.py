import argparse
import os

import torch
from utils import set_seed, collate_fn, collate_fn_student
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel, BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import numpy as np
from utils import score,print_table
import math
import pickle as pkl
from apex import amp
from prepro import read_semeval, read_tacred
from model import Bert4SemiRE2, DistillKL
import shutil
import time
import ipdb

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# rel_list = {'no_relation', 'Entity-Destination(e1,e2)', 'Cause-Effect(e2,e1)', 'Member-Collection(e2,e1)',
#              'Entity-Origin(e1,e2)', 'Message-Topic(e1,e2)', 'Component-Whole(e2,e1)',
#              'Component-Whole(e1,e2)', 'Instrument-Agency(e2,e1)', 'Product-Producer(e2,e1)',
#              'Content-Container(e1,e2)', 'Cause-Effect(e1,e2)', 'Product-Producer(e1,e2)',
#              'Content-Container(e2,e1)', 'Entity-Origin(e2,e1)', 'Message-Topic(e2,e1)',
#              'Instrument-Agency(e1,e2)', 'Member-Collection(e1,e2)', 'Entity-Destination(e2,e1)'}

dev_f1_iter, dev_pr_iter,dev_re_iter, test_f1_iter, test_pr_iter, test_re_iter = [], [], [], [], [], []

def softmax(x):
    x_max = x.max(axis=1)
    x_max = x_max.reshape([x.shape[0],1])
    x = (x - x_max) / 2.4
    x_exp = np.exp(x)
    softmax = x_exp / (x_exp.sum(axis=1).reshape([x.shape[0],1]))
    return softmax


def one_iter(args, t_features, inf_feature):
    right_num = 0
    neg_num = 0
    new_feature = []
    new_inf_feature = []
    for f in inf_feature:
        logits_list = [f['teacher_logits_{}'.format(i)] for i in range(args.model_num)]
        # logits_list = [f['teacher_logits_{}'.format(i)] for i in range(2,3)]
        label = list(set([np.argmax(logits) for logits in logits_list]))
        if len(label)==1:
            if f['relations']==label[0]:
                right_num+=1
            if f['relations'] == 0:
                neg_num+=1
            f['relations'] = label[0]
            new_feature.append(f)
        else:
            new_inf_feature.append(f)
    t_features.extend(new_feature)
    print('Consent features length:',len(t_features))
    print('neg:',neg_num)
    return t_features, new_inf_feature

def select_samples(preds, confidence, golds, ids, train_features, infer_features, k_samples):
    pred_conf = zip(preds, golds, confidence, ids)
    ranking = sorted(pred_conf, key=lambda x:x[2], reverse=True)[:k_samples]
    select_id = [rank[3] for rank in ranking]
    guess, gold = [rank[0] for rank in ranking],[rank[1] for rank in ranking]
    print('Infer on infer_data_set:')
    score(gold,guess,tag='infer')
    new_infer_features = []
    for i,feature in enumerate(infer_features):
        if feature['id'] in select_id:
            feature['relations'] = guess[select_id.index(feature['id'])]
            train_features.append(feature)
        else:
            new_infer_features.append(feature)

    return train_features, new_infer_features

def train_student(args, t_features, inf_feature):
    right_num = 0
    new_feature = []
    new_inf_feature = []
    for f in inf_feature:
        logits_list = [f['teacher_logits_{}'.format(i)] for i in range(args.model_num)]
        # logits_list = [f['teacher_logits_{}'.format(i)] for i in range(2,3)]
        label = list(set([np.argmax(logits) for logits in logits_list]))
        if len(label)==1:
            if f['relations']==label[0]:
                right_num+=1
            f['relations'] = label[0]
            new_feature.append(f)
        else:
            new_inf_feature.append(f)
    t_features.extend(new_feature)
    return t_features, new_inf_feature

def train(args, model, train_features, infer_features, dev_features, test_features, adj_list=None, id_list=None):
    def finetune(features, optimizer, num_epoch, num_steps, num=0):
        best_score = -1
        if args.is_student:
            train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn_student,
                                          drop_last=True)
        else:
            train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True,collate_fn=collate_fn,
                                          drop_last=True)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        for epoch in train_iterator:
            model.zero_grad()
            for step, batch in enumerate(train_dataloader):
                model.train()
                #input_ids, input_mask, subj_pos, obj_pos, relations, id, train_len
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'subj_pos': batch[2],
                          'obj_pos': batch[3],
                          'relations': torch.tensor(batch[4], dtype=torch.long).to(args.device),
                          }
                # if not args.use_pos:
                #     inputs['pos_attention'] = None
                logits, loss, _ = model(**inputs)

                if args.is_student:
                    if args.use_weight:
                        t_logits_list = torch.tensor(np.stack(batch[7]).squeeze()).transpose(0, 1).to(args.device)
                        loss_weight = batch[8].to(args.device)
                        for i in range(args.model_num):
                            loss += args.teacher_weight * soft_div(logits, t_logits_list[i].squeeze(),loss_weight[:,i])
                    else:
                        t_logits_list = torch.tensor(np.stack(batch[7]).squeeze()).transpose(0, 1).to(args.device)
                        for i in range(args.model_num):
                            loss += args.teacher_weight * soft_div(logits, t_logits_list[i].squeeze())#, soft_label_mask=soft_label_mask

                loss = loss / args.gradient_accumulation_steps

                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        # torch.nn.utils.clip_grad_norm_(optimizer_grouped_parameters, args.max_grad_norm)
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    scheduler.step()
                    optimizer.step()
                    model.zero_grad()
                    num_steps += 1
                if (step + 1) == len(train_dataloader) - 1 or (
                        args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                    pr, re, f1 = evaluate(args, model, dev_features, tag="dev")
                    if f1 > best_score:
                        best_score = f1
                        p, r, f = evaluate(args, model, test_features, tag="test")
                        if args.temp_path != "" and not args.is_student:
                            torch.save(model.state_dict(), args.temp_path.format(args.labeled_ratio, num))

        return pr,re,f1,p,r,f

    def infer(features):
        train_dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=True, collate_fn=collate_fn,
                                      drop_last=True)
        preds, confidence, golds, ids = [], [], [], []
        for epoch, batch in enumerate(train_dataloader):
            model.eval()
            #input_ids, input_mask, subj_pos, obj_pos, relations, id
            golds.extend(batch[4])
            ids.extend(batch[5])
            inputs = {'input_ids': batch[0].to(args.device),
                      'attention_mask': batch[1].to(args.device),
                      'subj_pos': batch[2],
                      'obj_pos': batch[3],
                      }
            # if not args.use_pos:
            #     inputs['pos_attention'] = None
            prediction, confid = model.predict(**inputs)
            preds.extend(prediction)
            confidence.extend(confid)
        return preds, confidence, golds, ids

    new_layer = ["extractor", "bilinear"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": 1e-4},
    ]
    soft_div = DistillKL(args.temperature)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    num_steps = 0
    set_seed(args)
    k_samples = math.ceil(len(infer_features) * args.data_ratio)
    num_iters = math.ceil(1.0 / args.data_ratio)
    model.zero_grad()
    for num in range(num_iters):
        pr,re,f1,p,r,f = finetune(train_features, optimizer, args.num_train_epochs, num_steps, num)
        dev_pr_iter.append(pr)
        dev_re_iter.append(re)
        dev_f1_iter.append(f1)
        test_pr_iter.append(p)
        test_re_iter.append(r)
        test_f1_iter.append(f)
        # if num==7:
        #     break
        if args.is_student:
            # num_iters=2
            # train_features, infer_features = select_samples(preds, confidence, golds, ids, train_features,
            #                                                 infer_features, len(infer_features))
            # pr, re, f1, p, r, f = finetune(train_features, optimizer, args.num_train_epochs, num_steps, num)
            # dev_pr_iter.append(pr)
            # dev_re_iter.append(re)
            # dev_f1_iter.append(f1)
            # test_pr_iter.append(p)
            # test_re_iter.append(r)
            # test_f1_iter.append(f)
            break
        else:
            preds, confidence, golds, ids = infer(infer_features)
            train_features, infer_features = select_samples(preds, confidence, golds, ids, train_features, infer_features, k_samples)
    if not args.is_student:
        max_index = dev_f1_iter.index(max(dev_f1_iter))
        shutil.move(args.temp_path.format(args.labeled_ratio, max_index), args.save_path.format(args.labeled_ratio, args.teacher_num))
        try:
            for i in range(num_iters):
                os.remove(args.temp_path.format(args.labeled_ratio, max_index))
        except:
            pass


def evaluate(args, model, features, tag="dev"):
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
                  }

        # if not args.use_pos:
        #     inputs['pos_attention'] = None
        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = np.argmax(preds, 1).tolist()
    prec_micro, recall_micro, f1_micro = score(golds, preds, tag=tag)

    return prec_micro, recall_micro, f1_micro

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="../Bert4SemiRE/dataset/semeval/", type=str)
    parser.add_argument("--feature_dir", default="./features_{}.pkl", type=str)

    parser.add_argument("--model_name_or_path", default="../Bert4SemiRE/bert/BERT-BASE-UNCASED",type=str)

    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument('--labeled_ratio', type=float, default=0.1)
    parser.add_argument('--unlabeled_ratio', type=float, default=0.5)

    parser.add_argument("--train_batch_size", default=20, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=20, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_class", type=int, default=19, help="Number of relation types in dataset.")

    parser.add_argument("--teacher_num", default=0, type=int, help="The teacher num.")
    parser.add_argument("--temperature", type=float, default=2.4, help="Distilling Temperature!")
    parser.add_argument("--teacher_weight", type=float, default=0.3, help="weight of mult teacher loss!")
    parser.add_argument("--model_num", default=2, type=int, help="The teacher num.")
    parser.add_argument("--save_path", default="./saved_models/{}/Bert4SemiRE_model_teacher_{}.pt", type=str)
    parser.add_argument("--temp_path", default="./saved_models/{}/temp/Bert4SemiRE_model_{}.pt", type=str)
    parser.add_argument("--load_path", default="", type=str)

    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed for initialization")

    parser.add_argument("--teacher_path", default="./saved_models/{}/Bert4SemiRE_model_teacher_{}.pt", type=str)

    parser.add_argument('--data_ratio', type=float, default=0.1)
    parser.add_argument('--use_pos',action='store_true', default=False)
    parser.add_argument('--use_adj',action='store_true', default=False)
    parser.add_argument('--is_student',action='store_true', default=False)
    parser.add_argument('--use_weight',action='store_true', default=False)
    parser.add_argument('--cuda', type=bool, default=True)

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_class,)
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path,)
    model = BertModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

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
    if args.is_student:
        with open(args.feature_dir.format(args.labeled_ratio), 'rb') as f:
            train_features, infer_features, dev_features, test_features, adj_list, id_list = pkl.load(f)
        train_features, infer_features = one_iter(args, train_features, infer_features)
    start = time.process_time()
    model = Bert4SemiRE2(config, model)
    # model.load_state_dict(torch.load(args.teacher_path.format(args.labeled_ratio, 0)))
    model.to(0)
    train(args, model, train_features, infer_features, dev_features, test_features)
    end = time.process_time()

    n_trainable_params, n_nontrainable_params = 0, 0
    for p in model.parameters():
        n_params = torch.prod(torch.tensor(p.shape))
        if p.requires_grad:
            n_trainable_params += n_params
        else:
            n_nontrainable_params += n_params
    print(
        'n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
    print('Running time: %s Seconds' % (end - start))
    torch.save(model.state_dict(), "./saved_models/{}/Bert4SemiRE_model_one_teacher.pt".format(args.labeled_ratio))

    print_table(dev_pr_iter, dev_re_iter, dev_f1_iter, test_pr_iter, test_re_iter, test_f1_iter, header='Best dev and test F1 with seed=%s:' % args.seed)

if __name__ == "__main__":
    main()