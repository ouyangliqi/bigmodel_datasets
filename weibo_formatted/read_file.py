import json
import logging
import os
from typing import List


def name2id(names_map: dict, relations_map: List[dict]):
    inverse_names_map = {name: nid for nid, name in names_map.items()}
    new_relations_map = relations_map
    count = 0
    for ind, conv in enumerate(new_relations_map):
        if 'child_info' in conv:
            for cid, child in enumerate(conv['child_info']):
                if child['@']:
                    for a_ind, user in enumerate(conv['child_info'][cid]['@']):
                        if user in inverse_names_map:
                            new_relations_map[ind]['child_info'][cid]['@'][a_ind] = inverse_names_map[user]
                            count += 1
    print("Changed user name: {}".format(count))
    return new_relations_map

def check_number(s):
    try:
        float(s)
        return True
    except:
        return False


def read_file(fpath):
    """Output format reference: ConvAI2 competition"""
    blogs = []
    with open(fpath) as f:
        tmp = json.load(f)
    for blog in tmp:
        # keys: ['names_map', 'contents_map', 'relations_map']
        conversations = []
        relations_map = name2id(blog['names_map'], blog['relations_map'])
        for conv in relations_map:
             # list of list
            if "child_info" in conv:
                # check if there are multi-users and reformat the name into person 1 and person 2
                cur_info = conv['root_info']
                first = {
                    "text_id": cur_info["text"],
                    "text": blog['contents_map'][str(cur_info["text"])],
                    "time": cur_info['time'],
                    "person_id": cur_info['name'],
                    "person": blog['names_map'][str(cur_info['name'])]
                }
                tmp_conv = []
                for child in conv['child_info']:
                    tmp_conv.append({
                        "text_id": child["text"],
                        "text": blog['contents_map'][str(child["text"])],
                        "time": child['time'],
                        "person_id": cur_info['name'],
                        "person": blog['names_map'][str(child['name'])]
                    })
                if tmp_conv:
                    first["children"] = tmp_conv
                    conversations.append(first)

        if conversations:
            blogs.append(conversations)
    print("GET conversations number: {}".format(len([c for blog in blogs for c in blog])))
    return blogs

def save_file(convs, filename):
    if convs:
        with open(filename, 'w') as f:
            json.dump(convs, f, ensure_ascii=False)

def main_filter(fpath, out_dir):
    try:
        res = read_file(fpath)
    except:
        logging.exception('Got exception on main handler')

    output = os.path.basename(fpath).split('.')[0] + '_original.json'
    save_file(res, os.path.join(out_dir, output))
