import os
import yaml
import pandas as pd
import numpy as np
import clip
import torch
import nltk
from nltk.stem import WordNetLemmatizer


def get_templates(dataset_name):
    label_file = f"data/{dataset_name}/classes_label_{dataset_name}.yml"
    with open(label_file, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    # classes = data['classes']  # list: ['brush hair', ...]
    # templates = data['templates']  # list: ['a video of a person {}.', ...]
    # obj_templates = data['obj_templates']
    return data
def get_xprompt(dataset_name):
    xprompt_file = f"data/{dataset_name}/classes_xprompt_{dataset_name}.yml"
    with open(xprompt_file, 'r') as f:
        text = yaml.load(f, Loader=yaml.FullLoader)
    return text

def get_ASKG_entity(dataset_name, classes, entity_type):
    ASKG_file = f"data/{dataset_name}/classes_ASKG_{dataset_name}.yml"
    with open(ASKG_file, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    en_dict = {} # {'c1':[obj1, obj2, ...], 'c2':[obj1, obj2, ...], ...]}
    # print(3,classes)
    if entity_type == 'o':
        en_list = [o for label, info in data.items() for o in info['obj_en_li']]
        # print(0,en_list)
        en_dict = {i: info['obj_en_li'] for i, (label, info) in enumerate(data.items())}
        # singularize nouns
        # nltk.download('wordnet')
        lemmatizer = WordNetLemmatizer()
        en_list = [lemmatizer.lemmatize(word, pos='n') for word in en_list]
        en_list = list(set(en_list))
        for i, li in en_dict.items():
            en_dict[i] = [lemmatizer.lemmatize(word, pos='n') for word in li]

    elif entity_type == 'a':
        en_list = [a for label, info in data.items() for a in info['sub_act_en_li']]
        en_list = list(set(en_list))
        en_dict = {i: info['sub_act_en_li'] for i, (label, info) in enumerate(data.items())}
    else:
        raise ValueError(f'entity_type should be o or a, but got {entity_type}')
    return en_list, en_dict

def expand_cls_tokenized(cls_tokenized, num_prompt):
    expanded_cls_tokenized = []
    for cls in cls_tokenized:
        cur_size = cls.size(0)
        if cur_size < num_prompt:
            repeats = num_prompt // cur_size
            expanded_cls = torch.cat([cls] * repeats, dim=0)
            remaining = num_prompt % cur_size
            if remaining > 0:
                expanded_cls = torch.cat([expanded_cls, cls[:remaining]], dim=0)

        else:
            expanded_cls = cls[:num_prompt]
        expanded_cls_tokenized.append(expanded_cls)
    expanded_cls_tokenized = torch.stack(expanded_cls_tokenized, dim=0)
    return expanded_cls_tokenized

def expand_cls_text(cls_text_list):
    min_prompt = min([len(i) for i in cls_text_list])
    max_prompt = max([len(i) for i in cls_text_list])

    expanded_cls_text_list = [sublist + [sublist[i % len(sublist)] for i in range(max_prompt-len(sublist))] for sublist in cls_text_list]
    return expanded_cls_text_list, min_prompt, max_prompt
def text_prompt_old(data):
    # text_aug = ['{}']
    text_aug = ['a video of a person {}.']

    text_dict = {}
    num_text_aug = len(text_aug)

    for ii, txt in enumerate(text_aug):
        text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for i, c in data.classes])

    classes = text_dict[0]

    return classes, num_text_aug, text_dict

def text_prompt(data, dataset: str, num_templates: int, cls_prompt_type: str):
    text = get_xprompt(dataset)
    templates = get_templates(dataset)['templates'] # k: classes, templates, obj_templates
    classes = [i[1] for i in data.classes] # ["c1", "c2", ...]
    num_classes = len(classes)
    n_prompts = [0, 0]

    total_templates = len(templates)
    num_templates = min(num_templates, total_templates)
    templates = templates[:num_templates] # a video of a person {}.

    tokenized_dict = {}
    cls_text_dict = {}
    text_dict = {}
    xoo_dict = {} # {0: [xprompt_oo,...], 1: [xprompt_oo], ...}
    xao_dict = {} # {0: [xprompt_ao,...], 1: [xprompt_ao], ...}
    xaa_dict = {} # {0: [xprompt_aa,...], 1: [xprompt_aa], ...}
    for i, t in enumerate(text.values()):
        xoo_dict[i] = t['xprompt_oo']
        xao_dict[i] = t['xprompt_ao']
        xaa_dict[i] = t['xprompt_aa']
    for ii, txt in enumerate(templates):
        text_dict[ii] = {
            'a': [[txt.format(c)] for i, c in enumerate(classes)],
            'xoo': [],
            'xao': [],
            'xaa': []
        }
        tokenized_dict[ii] = []
        for i, c in enumerate(classes):
            ci_xoo_list = text_dict[ii]['a'][i][:] # ["a {ci}."]
            ci_xao_list = text_dict[ii]['a'][i][:]
            ci_xaa_list = text_dict[ii]['a'][i][:]
            for j, t in enumerate(xoo_dict[i]):
                ci_xoo_list.append(txt.format(f"{c}, {t}")) # ["a video of a person brush hair, where hair..."]
            text_dict[ii]['xoo'].append(ci_xoo_list)
            for j, t in enumerate(xao_dict[i]):
                ci_xao_list.append(txt.format(f"{c}, {t}"))
            text_dict[ii]['xao'].append(ci_xao_list)
            for j, t in enumerate(xaa_dict[i]):
                ci_xaa_list.append(txt.format(f"{c}, {t}"))
            text_dict[ii]['xaa'].append(ci_xaa_list)
        if cls_prompt_type == 'xoo':
            cls_text_dict[ii] = text_dict[ii]['xoo']
        elif cls_prompt_type == 'xao':
            cls_text_dict[ii] = text_dict[ii]['xao']
        elif cls_prompt_type == 'xaa':
            cls_text_dict[ii] = text_dict[ii]['xaa']
        elif cls_prompt_type == 'xmix':
            cls_text_dict[ii] = []
            for i in range(len(classes)):
                cls_text_dict[ii].append(list(set(text_dict[ii]['xoo'][i] + text_dict[ii]['xao'][i] + text_dict[ii]['xaa'][i])))
        else:
            cls_text_dict[ii] = text_dict[ii]['a']


        cls_text_dict[ii], n_prompts[0], n_prompts[1] = expand_cls_text(cls_text_dict[ii])

        for n in range(n_prompts[1]):
            tokenized_dict[ii].append(torch.cat([clip.tokenize(t[n]) for t in cls_text_dict[ii]])) # [c 77, c 77,...]

        tokenized_dict[ii] = torch.cat(tokenized_dict[ii])
        # for i in range(len(cls_text_dict[ii])):
        #     tokenized_dict[ii].append(torch.cat([clip.tokenize(t) for t in cls_text_dict[ii][i]]))


    # cls_tokenized = [torch.cat([tokenized_dict[i][j] for i in range(num_templates)], dim=0) for j in range(num_classes)]

    cls_tokenized = torch.cat([v for v in tokenized_dict.values()]) # (num_templates max_prompt num_cls) 77


    # expand and repeat cls_tokenized dim to max
    # cls_tokenized = expand_cls_tokenized(cls_tokenized, max_prompt) # c max 77
    return cls_tokenized, cls_text_dict, tokenized_dict, num_templates, n_prompts


def entity_prompt(data, dataset: str, num_templates: int, entity_type: str):
    classes = [i[1] for i in data.classes] # ["c1", "c2", ...]
    templates = get_templates(dataset)
    if entity_type == 'o':
        templates = templates['obj_templates']
    elif entity_type == 'a':
        templates = templates['templates']
    else:
        raise ValueError(f'entity_type should be o or a, but got {entity_type}')
    total_templates = len(templates)
    num_templates = min(num_templates, total_templates)
    templates = templates[:num_templates]

    en_list, en_dict = get_ASKG_entity(dataset, classes, entity_type) # ['e1', 'e2', ...], {0:['','',...], 1:[], ...}


    classes_en_file = f"data/{dataset}/classes_{entity_type}_label_{dataset}.yml"
    if not os.path.isfile(classes_en_file):
        en_label_dict, en_label_list = get_en_labels(en_list, en_dict)
        data = {}
        data['list'] = en_list
        data['labels'] = {classes[k]: v for k, v in en_label_dict.items()}
        data['label_list'] = en_label_list
        with open(classes_en_file, 'w') as f:
            yaml.dump(data, f)
    else:
        with open(classes_en_file) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        en_label_list = data['label_list']

    en_text_list = [e for e in en_dict.values()]
    # print(0,en_text_list)
    en_text_list,_,num = expand_cls_text(en_text_list)
    # print(1,en_text_list)
    en_tokenized_dict = {}
    for ii, txt in enumerate(templates):
        en_tokenized_dict[ii] = []
        for n in range(num):
            en_tokenized_dict[ii].append(torch.cat([clip.tokenize(txt.format(t[n])) for t in en_text_list]))
        en_tokenized_dict[ii] = torch.stack(en_tokenized_dict[ii]) # e c 77
    en_tokenized = torch.stack([v for v in en_tokenized_dict.values()]) # num_templates max_en num_cls 77
    # print(entity_type,en_tokenized.shape)

    return en_tokenized, num_templates, en_label_list

def get_en_labels(en_list, en_dict):
    # classes = [i[1] for i in data.classes] # ["c1", "c2", ...]
    # en_list, en_dict = get_ASKG_entity(dataset, classes, entity_type) # ['e1', 'e2', ...], {0:['','',...], 1:[], ...}
    en_label_dict = {}
    en_label_list = []
    for c, en_li in en_dict.items():
        en_labels = []
        for i, en in enumerate(en_list):
            if en in en_li:
                en_labels.append(i)
        en_label_dict[c] = en_labels
        en_label_list.append(en_labels)
    return en_label_dict, en_label_list

if __name__ == '__main__':
#    text_prompt(dataset='ucf101', num_templates=8, cls_prompt_type='xmix')
    cls_tokenized, cls_text_dict, text_dict, n_templates, n_prompts = text_prompt(train_data, config.data.dataset, num_templates=8, cls_prompt_type='xmix')
    n_classes = int(cls_tokenized.size(0) / (n_templates * n_prompts[1]))
    num_text_aug = n_prompts[1] * n_templates
    cls_tokenized = rearrange(cls_tokenized, '(x a) d -> x a d', a=n_classes)
    x, a, d = cls_tokenized.size()
    clip_model.eval()
    with torch.no_grad():
    classes_features = torch.stack([clip_model.encode_text(cls_tokenized[i].squeeze()) for i in range(x)])
      # classes_features = clip_model.encode_text(cls_tokenized)
      # print(classes_features.shape)
    classes_features = classes_features.permute(1, 0, 2)  # a x d
    classes_features = classes_features.to('cpu')
    torch.save(classes_features, classes_feats_file)

   

