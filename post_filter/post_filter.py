import json
import os
import re
import sys
import time

from joblib import Parallel, delayed
from line_profiler import LineProfiler

from chunkify import parallel_apply_line_by_line

sys.path.append(os.path.abspath(".."))

import logging

from cleantext import clean
from util.langconv import *
from util.utils import *

from regexes import fixup
from remove_codes import remove_codes

logger = None


def setup_logger(name_logfile, path_logfile):
    path_logfile = path_logfile + name_logfile
    logger = logging.getLogger(name_logfile)
    formatter = logging.Formatter('[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s]: %(message)s')
    fileHandler = logging.FileHandler(path_logfile, mode='w')
    fileHandler.setFormatter(formatter)
    # streamHandler = logging.StreamHandler()
    # streamHandler.setFormatter(formatter)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(fileHandler)
    # logger.addHandler(streamHandler)

    logger.info(path_logfile)
    return logger

def traditionalTosimplified(content):
    line = Converter("zh-hans").convert(content)
    return line

def remove_emoji3(text):
    emoji_regex = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002500-\U00002BEF"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"
        "\u3030"
        "\\*\u20e3"
        "#\u20e3"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_regex.sub(r"", text)
    return text.strip()


def is_rare_word(word):
    try:
        word.encode("gb2312")
    except:
        return True
    return False


def remove_unprintable_char(s):
    # pre_remove = s[:]
    s = re.sub(r"\\u.{4}", "", s)
    new_s = ""
    unprintable = ""
    for word in s:
        if is_rare_word(word):
            unprintable += word
            continue
        else:
            new_s += word
    # if pre_remove != new_s:
    #     logger.info("before remove unprintable: {}".format(pre_remove))
    #     logger.info("remove_unprintable_char: {}".format(unprintable))
    return new_s


def cleantext_clean(text):
    text = clean(
        text,
        fix_unicode=True,
        to_ascii=False,
        normalize_whitespace=True,
        no_line_breaks=True,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        replace_with_url="",
        replace_with_email="",
        replace_with_phone_number="<PHONE>")
    return text


def blacklist(s, cleaner):
    dirty_words = cleaner.processor.extract_keywords(s)
    if len(dirty_words) > 0 and dirty_words[0].strip():
        # logger.info("dirty_words: {}".format(dirty_words))
        return True
    return False


def merge_prediction_result(fastext_model, textcnn_model, cont):
    flag = True

    label_1, pro_1 = fastext_model.predict(cont)
    label_2, pro_2 = textcnn_model.predict(cont)

    if label_1 == label_2:
        if label_1[0] == "__label__0":
            flag = True
        else:
            flag = False
    else:
        if label_2 == 2 and label_1[0] != "__label__2":
            flag = True
        elif ((label_2 == 0 or label_2 == 3 or label_2 == 5 or label_2 == 4) and pro_2 >= 0.5):
            flag = True
        else:
            flag = False
    return flag, label_1[0], label_2


def check_if_doc_valid(content, cleaner, fastext_model, textcnn_model):
    valid = []
    for cont in content:
        if blacklist(cont, cleaner):
            logger.info("unvalid content: {}".format(cont))
            return []
        flag, label_1, label_2 = merge_prediction_result(fastext_model, textcnn_model, cont)
        if flag:
            if label_2 != 4:
                valid.append(cont)
            else:
                logger.info("remove unvalid adv {}".format(cont))
                return []
        else:
            logger.info("unvalid content detected by model: {}, fasttext_label {}, textcnn_label {}".format(cont, label_1, label_2))
    return valid


def clean_up(text):
    fixed_text = remove_unprintable_char(text)
    fixed_text = remove_codes(fixed_text)
    fixed_text = cleantext_clean(fixed_text)
    fixed_text = traditionalTosimplified(fixed_text)
    return fixed_text


def check_if_doc_valid_only1model(content, cleaner, model, model_type):
    valid = []
    if blacklist(" ".join(content), cleaner):
        logger.info("unvalid content: {}".format(" ".join(content)))
        return []

    labels, _ = model.batch_predict(content)
    for cont, label in zip(content, labels):
        if model_type == "fasttext": label = int(label[0][-1])
        if label in [0, 4, 3, 5]:
            if len(cont) <= 4:
                continue
            if label != 4:
                valid.append(clean_up(cont))
            else:
                logger.info("remove unvalid adv {}".format(cont))
        else:
            logger.info("unvalid content detected by model: {}, {}_label {}".format(cont, model_type, label))
            return []
    return valid


def process_line(line, fasttext_model, fn, cleaner):
    doc = json.loads(line[:-1])
    original_length = len(doc["cont"])
    lines_keep = []

    if len(" ".join(doc['cont'])) < 10:
        return None, original_length

    valid = check_if_doc_valid_only1model(doc["cont"], cleaner, fasttext_model, "fasttext")
    for text in valid:
        fixed_text = fixup(text, fn)
        if len(fixed_text) > 1:
            lines_keep.append(fixed_text)

    if len(lines_keep) > 0:
        doc["cont"] = lines_keep
    else:
        return None, original_length
    return doc, original_length


def main_filter(fn: str, out_dir):
    input_dir = os.path.dirname(fn)

    pos = fn.rfind(".")
    fn_out = fn[:pos] + "_filter" + fn[pos:]
    fn_out = fn_out.replace(
            input_dir,
            out_dir
        )

    log_path = os.path.join(os.path.dirname(fn_out), 'log/')
    global logger
    logger = setup_logger(os.path.basename(fn) + 'processed.log', log_path)


    cleaner = Cleaner()
    # cleaner.load_black_list(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "blacklist/post_blacklist/data/offensive/**"))
    cleaner.load_black_list(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "blacklist/post_blacklist/only/**"))

    fasttext_model = TextAuditing()

    orig_doc_line = 0
    res_doc_line = 0

    res = []
    res, orig_doc_line = parallel_apply_line_by_line(fn, 20, process_line, [fn, cleaner])
    # with open(fn, "r") as f:
    #     for line in f.readlines():
    #         doc = json.loads(line[:-1])
    #         lines_keep = []

    #         orig_doc_line += len(doc['cont'])

    #         valid = check_if_doc_valid_only1model(doc["cont"], cleaner, fasttext_model, "fasttext")
    #         for doc_line in valid:
    #             fixed_text = fixup(doc_line, fn)
    #             if len(fixed_text) > 1:
    #                 lines_keep.append(fixed_text)


    #         if len(lines_keep) > 0:
    #             doc["cont"] = lines_keep
    #             res_doc_line += len(lines_keep)
    #             res.append(doc)

        # res = Parallel(n_jobs=8, backend="threading")(delayed(process_line)(fn, line, cleaner, fasttext_model, orig_doc_line) for line in f.readlines()[:100])
        # res_doc_line = sum([len(doc["cont"]) for doc in res])

    if res:
        fp_out = open(fn_out, "w")
        for doc in res:
            if len(doc['cont']) > 0:
                res_doc_line += len(doc['cont'])
                fp_out.write(json.dumps(doc, ensure_ascii=False) + "\n")
        fp_out.close()
    print("orig_doc_line: {}, res_doc_line: {}, file: {}".format(orig_doc_line, res_doc_line, fn))



if __name__ == "__main__":
    # lprofiler = LineProfiler(main_filter, check_if_doc_valid_only1model)
    start = time.time()
    # lprofiler.enable()
    main_filter(sys.argv[1], sys.argv[2])
    end = time.time()
    print("time: {}".format(end - start))
    # lprofiler.disable()
    # lprofiler.print_stats()



# for line in f:
#     doc = json.loads(line[:-1])
#     lines_keep = []
#     orig_doc_line += len(doc["cont"])

#     valid = check_if_doc_valid_only1model(doc["cont"], cleaner, fasttext_model, "fasttext")
#     for doc_line in valid:
#         fixed_text = fixup(doc_line, fn)
#         if len(fixed_text) > 1:
#             lines_keep.append(fixed_text)


#     res_doc_line += len(lines_keep)
#     if len(lines_keep) > 0:
#         doc["cont"] = lines_keep
#         res.append(doc)


#266013
