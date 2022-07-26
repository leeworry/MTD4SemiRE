import json
from tqdm import tqdm
import ipdb

rel_to_id = {'no_relation': 0, 'Entity-Destination(e1,e2)': 1, 'Cause-Effect(e2,e1)': 2, 'Member-Collection(e2,e1)': 3,
             'Entity-Origin(e1,e2)': 4, 'Message-Topic(e1,e2)': 5, 'Component-Whole(e2,e1)': 6,
             'Component-Whole(e1,e2)': 7, 'Instrument-Agency(e2,e1)': 8, 'Product-Producer(e2,e1)': 9,
             'Content-Container(e1,e2)': 10, 'Cause-Effect(e1,e2)': 11, 'Product-Producer(e1,e2)': 12,
             'Content-Container(e2,e1)': 13, 'Entity-Origin(e2,e1)': 14, 'Message-Topic(e2,e1)': 15,
             'Instrument-Agency(e1,e2)': 16, 'Member-Collection(e1,e2)': 17, 'Entity-Destination(e2,e1)': 18}
rel = ['no_relation', 'org:founded_by', 'per:spouse', 'per:age', 'per:title', 'org:alternate_names', 'per:other_family',
       'org:top_members/employees', 'per:parents', 'org:subsidiaries', 'per:religion', 'org:city_of_headquarters',
       'per:cities_of_residence', 'org:parents', 'per:origin', 'org:stateorprovince_of_headquarters', 'per:charges',
       'org:website', 'per:stateorprovinces_of_residence', 'per:countries_of_residence', 'per:siblings',
       'per:employee_of', 'org:country_of_headquarters', 'per:stateorprovince_of_death', 'org:dissolved', 'org:members',
       'per:cause_of_death', 'org:founded', 'per:date_of_death', 'org:number_of_employees/members', 'org:member_of',
       'per:alternate_names', 'per:schools_attended', 'per:date_of_birth', 'per:children',
       'org:political/religious_affiliation', 'per:city_of_death', 'per:country_of_death', 'org:shareholders',
       'per:stateorprovince_of_birth', 'per:city_of_birth', 'per:country_of_birth']
VB_list=['VB','VBN','VBD','VBG','VBP','VBZ']#VBN:-ed,VBD:is
IN_list=['IN','RB','POS']
sents_list_sem = []
sents_list_tac = []


# with open('dataset/semeval/sent_dual.txt') as f:
with open('dataset/semeval/sent.txt') as f:
    lines = f.readlines()
    for line in lines:
        # if int(line.strip().split('\t')[0]) in case:
        #     print(line.strip().split('\t')[1])
        sents_list_sem.append(line.strip().split('\t')[1])

# with open('dataset/semeval/sent_dual.txt') as f:
#     lines = f.readlines()
#     for line in lines:
#         sents_list_sem.append(line.strip().split('\t')[1])
#with open('dataset/json_tac/tac_sent.json', 'r') as f:
#    sents_list_tac = json.load(f)

def process_data(data, tokenizer, max_seq_length=512):
    features = []
    for sample in tqdm(data, desc="Example"):
        sents = []
        sent_map = []
        subj_start, subj_end = sample['subj_start'], sample['subj_end']
        obj_start, obj_end = sample['obj_start'], sample['obj_end']

        for i, token in enumerate(sample['tokens']):
            tokens_wordpiece = tokenizer.tokenize(token)

            if i == subj_start:
                tokens_wordpiece = ['[entity1]'] + tokens_wordpiece
            if i == subj_end:
                tokens_wordpiece = tokens_wordpiece + ['[entity2]']
            if i == obj_start:
                tokens_wordpiece = ['[entity3]'] + tokens_wordpiece
            if i == obj_end:
                tokens_wordpiece = tokens_wordpiece + ['[entity4]']
            sent_map.append(len(sents))
            sents.extend(tokens_wordpiece)
        sent_map.append(len(sents))
        train_len = len(sents)
        subj_pos = [(sent_map[subj_start], sent_map[subj_end])]
        obj_pos = [(sent_map[obj_start], sent_map[obj_end])]
        sents = sents[:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
        feature = {'input_ids': input_ids,
                   'subj_pos': subj_pos,
                   'obj_pos': obj_pos,
                   'relations': rel_to_id[sample['relation']],
                   'token_type_ids': None,
                   'id': sample['id'],
                   }
        features.append(feature)

def read_semeval(file_train, tokenizer, max_seq_length=512):
    # original sentence
    features = []
    if file_train == "":
        return None
    with open(file_train, 'r') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
    for sample in tqdm(data, desc="Example"):
        pos_attention = []
        tokens = sents_list_sem[int(sample['id']) - 1]
        en1, en2 = min(sample['subj_end'],sample['obj_end']), max(sample['subj_start'],sample['obj_start'])
        for num, pos in enumerate(sample['stanford_pos'][en1:en2+1]):
            if pos in VB_list:
                pos_attention.append(num+en1)
            if pos in IN_list and (num+en1-1 in pos_attention or en2 - en1<3):
                pos_attention.append(num+en1)

        encoded_dict = tokenizer.encode_plus(
            tokens,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            padding=True,
            return_attention_mask=False,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
        try:
            # Find e1(id:2487) and e2(id:2475) position
            pos1 = (encoded_dict['input_ids'] == 2487).nonzero(as_tuple=False)[0][1].tolist()
            pos2 = (encoded_dict['input_ids'] == 2475).nonzero(as_tuple=False)[0][1].tolist()
            # Add the encoded sentence to the list.
            input_ids = encoded_dict['input_ids']
            # And its attention mask (simply differentiates padding from non-padding).

            feature = {'input_ids': input_ids.squeeze(),
                       'subj_pos': pos1,
                       'obj_pos': pos2,
                       'relations': rel_to_id[sample['relation']],
                       'id': sample['id'],
                       'pos_attention': pos_attention,
                       }
            features.append(feature)
        except:
            ipdb.set_trace()

    print("# of documents {}, # of documents {}.".format(len(data), len(features)))
    return features

def read_tacred(file_train, tokenizer, max_seq_length=512):
    # original sentence
    features = []
    if file_train == "":
        return None
    with open(file_train, 'r') as f:
        try:
            lines = f.readlines()
            data = [json.loads(line) for line in lines]
        except:
            ipdb.set_trace()

    for sample in tqdm(data, desc="Example"):
        pos_attention = []
        tokens = sents_list_tac[int(sample['id'])]
        en1, en2 = min(sample['subj_end'], sample['obj_end']), max(sample['subj_start'], sample['obj_start'])
        for num, pos in enumerate(sample['stanford_pos'][en1:en2 + 1]):
            if pos in VB_list:
                pos_attention.append(num + en1)
            if pos in IN_list and (num + en1 - 1 in pos_attention or en2 - en1 < 3):
                pos_attention.append(num + en1)
        encoded_dict = tokenizer.encode_plus(
            tokens,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            padding=True,
            return_attention_mask=False,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
        try:
            # Find e1(id:2487) and e2(id:2475) position
            pos1 = (encoded_dict['input_ids'] == 2487).nonzero(as_tuple=False)[0][1].tolist()
            pos2 = (encoded_dict['input_ids'] == 2475).nonzero(as_tuple=False)[0][1].tolist()
            # Add the encoded sentence to the list.
            input_ids = encoded_dict['input_ids']
            # And its attention mask (simply differentiates padding from non-padding).
            feature = {'input_ids': input_ids.squeeze(),
                       'subj_pos': pos1,
                       'obj_pos': pos2,
                       'relations': rel.index(sample['relation']),
                       'id': sample['id'],
                       'pos_attention': pos_attention,
                       }
            features.append(feature)
        except:
            ipdb.set_trace()

    print("# of documents {}, # of documents {}.".format(len(data), len(features)))
    return features