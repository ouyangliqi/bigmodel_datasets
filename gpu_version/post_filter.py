import json
import os
import re
import sys
import time
from multiprocessing import Manager, current_process
from multiprocessing.dummy import Manager

from chunkify import parallel_apply_line_by_line
from joblib import Parallel, delayed
from line_profiler import LineProfiler

sys.path.append(os.path.abspath(".."))

import logging

from cleantext import clean
from remove_codes import remove_codes
from util.langconv import *
from util.utils import *

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
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>")
    return text


def blacklist(s, cleaner):
    dirty_words = cleaner.processor.extract_keywords(s)
    if len(dirty_words) > 0 and dirty_words[0].strip():
        # logger.info("dirty_words: {}".format(dirty_words))
        return True
    return False


def clean_up(text):
    fixed_text = cleantext_clean(text)
    fixed_text = remove_unprintable_char(fixed_text)
    fixed_text_2 = remove_codes(fixed_text)

    # 重复模式匹配，删除连续重复三次或以上的模式，保留一次模式
    # print("3")
    # pattern = re.compile(r'(.+?)\1\1+')
    # text = pattern.sub(r'\1', fixed_text_2)
    # if fixed_text_2 != text:
    #     logger.info("remove repeat: {}".format(pattern.findall(fixed_text_2)))
    #     logger.info("remove repeat text: {}".format(fixed_text_2))

    # if fixed_text != fixed_text_2:
    #     logger.info("remove codes: {}".format(fixed_text))
    # fixed_text = traditionalTosimplified(text)
    return fixed_text_2


def check_if_doc_valid_only1model(content, cleaner, model, model_type, logger):
    valid = []
    for cont in content:
        if blacklist(" ".join(cont), cleaner):
            logger.info("unvalid content: {}".format(" ".join(content)))
            return []

    labels, _ = model.batch_predict(content)
    for cont, label in zip(content, labels):
        if model_type == "fasttext": label = int(label[-1])
        if label in [0, 4, 3]:
            if label != 4:
                valid.append(cont)
            # else:
            #     logger.info("remove unvalid adv {}".format(cont))
        else:
            logger.info("unvalid content detected by model: {}, {}_label {}".format(cont, model_type, label))
            return []
    return valid


def process_line(line, model, fn, cleaner, logger):
    # for multiprocessing
    doc = json.loads(line[:-1])
    original_length = len(doc["cont"])
    lines_keep = []

    if len(" ".join(doc['cont'])) < 10:
        return None, original_length

    valid = check_if_doc_valid_only1model(doc["cont"], cleaner, model, "fasttext", logger)
    for text in valid:
        cleaned_text = pro_process(text, fn)
        if cleaned_text is not None:
            lines_keep.append(cleaned_text)

    if len(lines_keep) > 0:
        doc["cont"] = lines_keep
    else:
        return None, original_length
    return doc, original_length


def pro_process(doc_line, fn):
    cleaned_text = clean_up(doc_line)
    if len(cleaned_text) > 1:
        return cleaned_text
    else:
        return None


def single_process(fn, cleaner, model, model_type):
    # for single processing
    orig_doc_line, res_doc_line = 0, 0
    res = []
    with open(fn, "r") as f:
        for line in f.readlines():
            doc = json.loads(line[:-1])
            lines_keep = []

            orig_doc_line += len(doc['cont'])

            valid = check_if_doc_valid_only1model(doc["cont"], cleaner, model, model_type)
            torch.cuda.empty_cache()
            # multi-threading
            if len(valid) > 0:
                fix_texts = Parallel(n_jobs=len(valid), prefer="threads")(delayed(pro_process)(doc_line, fn) for doc_line in valid)
                for r in fix_texts:
                    if r is not None:
                        lines_keep.append(r)


            if len(lines_keep) > 0:
                doc["cont"] = lines_keep
                res_doc_line += len(lines_keep)
                res.append(doc)
    return res, orig_doc_line, res_doc_line


def main_filter(fn: str, out_dir, queue=None):
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

    try:
        # single process
        start = time.time()

        if queue is not None:
            gpu_id = queue.get()
            ident = current_process().ident
            # TODO(chloe): if gpu is enough
            torch.cuda.set_device(gpu_id)
            logger.info('{}: starting process on GPU {}'.format(ident, gpu_id))

        cleaner = Cleaner()
        # cleaner.load_black_list(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "blacklist/post_blacklist/data/offensive/**"))
        cleaner.load_black_list(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "blacklist/post_blacklist/only/**"))

        orig_doc_line = 0
        res_doc_line = 0

        res = []

        model_type = "textcnn"
        textcnn = TextCNN()
        # model_type = "fasttext"
        # fasttext_model = TextAuditing()

        res, orig_doc_line, res_doc_line = single_process(fn, cleaner, textcnn, model_type)

        if model_type == "textcnn":
            del textcnn
            torch.cuda.empty_cache()

        # multiprocess
        # res, orig_doc_line = parallel_apply_line_by_line(fn, 20, process_line, [fn, cleaner, logger], loger_args=[os.path.basename(fn) + 'processed.log', log_path])

        if res:
            fp_out = open(fn_out, "w")
            for doc in res:
                if len(doc['cont']) > 0:
                    res_doc_line += len(doc['cont'])
                    fp_out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            fp_out.close()
        print("orig_doc_line: {}, res_doc_line: {}, file: {}".format(orig_doc_line, res_doc_line, fn))

        end = time.time()
        print("time: {}".format(end - start))
        logger.info("time: {}".format(end - start))
    except:
        logger.exception('Got exception on main handler')
    finally:
        if queue is not None:
            queue.put(gpu_id)


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    lprofiler = LineProfiler(main_filter, check_if_doc_valid_only1model, pro_process, single_process)
    start = time.time()
    lprofiler.enable()
    queue = Manager().Queue()
    queue.put(0)
    main_filter(sys.argv[1], sys.argv[2], queue=queue)
    end = time.time()
    print("time: {}".format(end - start))
    lprofiler.disable()
    lprofiler.print_stats()



