from datetime import datetime
from typing import List
import json


def time_compare(time1, time2):
    t1 = datetime.strptime(time1, '%Y-%m-%d-%H-%M-%S')
    t2 = datetime.strptime(time2, '%Y-%m-%d-%H-%M-%S')
    if t1 <= t2:
        return True
    else:
        return False


def check_conversation(relation):
    root_name = relation['root_info']['name']
    flag = False #  filter conversation only the speaker himself
    for child in relation['child_info']:
        if child['name'] != root_name:
            flag = True
    return flag


def generate_tree(nodes, depth, parent, visited):
    if depth > 30:
        return []
    tree = []
    for child in nodes:
        if child in visited:
            continue
        if child["@"] and child["@"][0] == parent and child["name"] != parent:
            visited.append(child)
            child["children"] = generate_tree(nodes, depth + 1,child["name"], visited)
            tree.append(child)
    return tree


def format2tree(conv)-> List:
    cur_info = conv['root_info']
    first = {
        "text": cur_info["text"],
        "time": cur_info['time'],
        "name": cur_info['name'],
        "children": []
    }
    sorted_child = sorted(conv['child_info'], key=lambda x: datetime.strptime(x['time'], '%Y-%m-%d-%H-%M-%S'))[:2000]
    root = {"person_id": "root", "children": [first]}

    for child in sorted_child:
        if child["@"]:
            pass
        else:
            child["@"] = [first["name"]]
    gc = generate_tree(sorted_child, 0, first["name"], [])
    first["children"] = gc
    return root


def hepler(node, used, path, res):
    if not node or not node['children']:
        res.append(path[:])
        return res

    for i in node['children']:
        formatted_i = {
            "text": str(i["text"]),
            "time": i["time"],
            "person_id": str(i["name"]),
        }
        if formatted_i in used:
            continue
        used.append(formatted_i)
        path.append(formatted_i)
        hepler(i, used, path, res)
        used.remove(formatted_i)
        path.pop()


def generate_res(node)->List:
    # DFS
    res = []
    hepler(node, [], [], res)
    return res


if __name__ == "__main__":
    relation ={
        "root_info":{
            "text": 1,
            "time": "2019-10-02-22-58-54",
            "name": 1,
            "@": []
        },
        "child_info":[
            {
                "text": 2,
                "time": "2019-10-02-23-08-54",
                "name": 2,
                "@": []
            },
            {
                "text": 3,
                "time": "2019-10-02-23-18-54",
                "name": 3,
                "@": [2]
            },
            {
                "text": 4,
                "time": "2019-10-02-23-28-54",
                "name": 4,
                "@": []
            },
            {
                "text": 5,
                "time": "2019-10-02-23-38-54",
                "name": 5,
                "@": [2]
            },
            {
                "text": 6,
                "time": "2019-10-02-23-48-54",
                "name": 6,
                "@": [3]
            },
            {
                "text": 7,
                "time": "2019-10-02-23-48-54",
                "name": 7,
                "@": [1]
            },
            {
                "text": 8,
                "time": "2019-10-02-23-48-54",
                "name": 1,
                "@": [7]
            },
        ]
    }

    root = format2tree(relation)
    print(json.dumps(root, ensure_ascii=False))
    res = generate_res(root)
    print()
    print(json.dumps(res, ensure_ascii=False))
