import os
import sys
import random
import numpy as np
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import torch

NO_RELATION = 0

def ensure_dir(d, verbose=True):
    if not os.path.exists(d):
        if verbose:
            print("Directory {} do not exist; creating...".format(d))
        os.makedirs(d)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def print_table(*args, header=''):
    print(header)
    for tup in zip(*args):
        print('\t'.join(['%.4f' % t for t in tup]))


def collate_fn(batch):
    input_ids = pad_sequence([f["input_ids"] for f in batch],batch_first=True)
    max_len = input_ids.shape[1]
    input_mask = [[1.0] * f["input_ids"].shape[0] + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    subj_pos = [f["subj_pos"] for f in batch]
    obj_pos = [f["obj_pos"] for f in batch]
    relations = [f["relations"] for f in batch]
    id = [f["id"] for f in batch]
    pos_attention = [f["pos_attention"] for f in batch]
    input_mask = torch.tensor(input_mask, dtype=torch.float)


    output = (input_ids, input_mask, subj_pos, obj_pos, relations, id, pos_attention)
    return output

def collate_fn_student(batch):
    model_num = 2
    input_ids = pad_sequence([f["input_ids"] for f in batch],batch_first=True)
    max_len = input_ids.shape[1]
    input_mask = [[1.0] * f["input_ids"].shape[0] + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    subj_pos = [f["subj_pos"] for f in batch]
    obj_pos = [f["obj_pos"] for f in batch]
    relations = [f["relations"] for f in batch]
    token_type_ids = [f["token_type_ids"] + [1] * (max_len - len(f["token_type_ids"])) for f in batch if "token_type_ids" in f]
    id = [f["id"] for f in batch]
    soft_logits = [[f['teacher_logits_{}'.format(str(m))] for m in range(model_num)] for f in batch]

    soft_label_mask =[np.ones(soft_logits[0][0].shape) for i in range(len(soft_logits))]
    for i,soft_l in enumerate(soft_logits):
        soft_label_mask[i][np.argmax(soft_l[0])]=0

    input_mask = torch.tensor(input_mask, dtype=torch.float)
    # loss_weight = torch.tensor([f["logits_weight"] for f in batch], dtype=torch.float)
    loss_weight = None
    if len(token_type_ids)==0:
        output = (input_ids, input_mask, subj_pos, obj_pos, relations, id, soft_label_mask, soft_logits, loss_weight)
    else:
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        output = (input_ids, input_mask, subj_pos, obj_pos, relations, id, token_type_ids, soft_logits, loss_weight)
    return output

def score(key, prediction, verbose=False, NO_RELATION=NO_RELATION, tag='train'):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]

        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1:
                sys.stdout.write(' ')
            if prec < 1.0:
                sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1:
                sys.stdout.write(' ')
            if recall < 1.0:
                sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1:
                sys.stdout.write(' ')
            if f1 < 1.0:
                sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
        print("")

    # Print the aggregate score
    if verbose:
        print("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(
            sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(
            sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print("SET NO_RELATION ID in {}: ".format(tag), NO_RELATION)
    print("Precision (micro): {:.3%}".format(prec_micro))
    print("   Recall (micro): {:.3%}".format(recall_micro))
    print("       F1 (micro): {:.3%}".format(f1_micro))
    return prec_micro, recall_micro, f1_micro