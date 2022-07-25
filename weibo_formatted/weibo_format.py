import json
import logging
import os
import sys
from typing import List

from Tree import *

base_dir = '/mnt/datadisk0/migo/crawler/weibo_crawler_scf/crawler_result/hot_results'

logger = None


def setup_logger(name_logfile, path_logfile):
    path_logfile = path_logfile + name_logfile
    logger = logging.getLogger(name_logfile)
    formatter = logging.Formatter("[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s]: %(message)s")
    fileHandler = logging.FileHandler(path_logfile, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)

    logger.info(path_logfile)
    return logger


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
                            assert new_relations_map[ind]['child_info'][cid]['@'][a_ind] == inverse_names_map[user]
    logger.info("Changed user name: {}".format(count))
    return new_relations_map


def check_conversation(mapping):
    root_name = mapping['root_info']['name']
    flag = False #  filter conversation only the speaker himself
    for child in mapping['child_info']:
        if child['name'] != root_name:
            flag = True
    return flag


def check_number(s):
    try:
        float(s)
        return True
    except:
        return False


def statistics(convs):
    # 统计两人对话及两人以上的对话
    twopersons = 0
    multipersons = 0
    # 两人对话并且轮数大于1
    suitable_convs = 0
    for conv in convs:
        if len(set([c['person'] for c in conv])) == 2:
            twopersons += 1
            if len(conv) > 2:
                suitable_convs += 1
        else:
            multipersons += 1
    return suitable_convs, twopersons, multipersons

def save_file(convs, filename):
    if convs:
        with open(filename, 'w') as f:
            json.dump(convs, f, ensure_ascii=False)


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
            res = [] # list of list
            if ("child_info" in conv) and check_conversation(conv):
                # check if there are multi-users and reformat the name into person 1 and person 2
                root = format2tree(conv)
                if root["children"]:
                    res = generate_res(root)
                logger.info("Completed Tree Building")

                # for each conversation in tmp_conv: check if only one specker
                # check if there are multi-users and reformat the name into person 1 and person 2
                valid_conv = []
                for cs in res:
                    if len(set([i['person_id'] for i in cs])) > 1:
                        tmp_cs = []
                        for c in cs:
                            tmp_cs.append({
                                "text": blog['contents_map'][c["text"]],
                                "time": c['time'],
                                "person_id": c['person_id'],
                                "person": blog['names_map'][c['person_id']]
                            })
                        valid_conv.append(tmp_cs)

                if valid_conv:
                    conversations.extend(valid_conv)

        if conversations:
            blogs.append(conversations)
    logger.info("GET conversations number: {}".format(len([c for blog in blogs for c in blog])))
    return blogs


def format_conversations(blogs):
    conversations = []
    for blog in blogs:
        for conv in blog:
            sender_mapping = {}
            aconv = []
            for cid, cs in enumerate(conv):
                if cs['person'] not in sender_mapping:
                    tmp_len = len(sender_mapping)
                    sender_mapping[cs['person']] = tmp_len + 1
                cs['id'] = cid
                cs['sender'] = 'Person {}'.format(sender_mapping[cs['person']])
                aconv.append({
                    'id': cid,
                    'sender': 'Person {}'.format(sender_mapping[cs['person']]),
                    'text': cs['text'],
                    'time': cs['time'],
                    'person_id': cs['person_id'],
                    'person': cs['person']
                })
            if aconv:
                conversations.append(aconv)
    logger.info("FORMAT conversations number: {}".format(len([c for blog in blogs for c in blog])))
    return conversations


def main_filter(fpath, out_dir):
    log_path = os.path.join(os.path.dirname(out_dir), 'log/')
    global logger
    logger = setup_logger(os.path.basename(fpath) + '.log', log_path)

    try:
        res = read_file(fpath)
        res = format_conversations(res)
    except:
        logger.exception('Got exception on main handler')

    a, b, c = statistics(res)
    logger.info("多轮两人对话占比: {:.2f}，两人对话数目: {}, 多人对话数目: {}".format(a/b if b!= 0 else a, b, c))
    output = os.path.basename(fpath).split('.')[0] + '_format.json'
    save_file(res, os.path.join(out_dir, output))
    # print(res)


if __name__ == "__main__":
    base_dir = "/mnt/datadisk0/migo/crawler/weibo_crawler_scf/crawler_result/hot_results/"
    out_dir = "/mnt/cfs/weibo_comments/formatted"
    # subdir = "crawler_result_2022-04-04-14-58-11.json"  # 六六大顺
    subdir = "crawler_result_2022-04-13-14-27-22.json"
    # [2022-05-29 21:31:32,453][weibo_format.py][line:177][INFO]: 多轮两人对话占比: 0.02，两人对话数目: 57475, 多人对话数目: 50
    main_filter(base_dir + subdir, out_dir)
