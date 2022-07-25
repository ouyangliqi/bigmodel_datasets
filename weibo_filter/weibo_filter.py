import collections
import hashlib
import json
import logging
import os
import re
import sys

sys.path.append(os.path.abspath(".."))
import unicodedata
from datetime import datetime
from typing import Type

import ftfy
import numpy as np
from cleantext import clean
from util.langconv import *
from util.utils import Blandness, Cleaner

HASH_TYPE: Type[np.uint64] = np.uint64
HASH_SIZE = HASH_TYPE(0).nbytes


REPLACE_CONT_BLANKS_RE = re.compile(" +")
logger = None


def setup_logger(name_logfile, path_logfile):
    path_logfile = path_logfile + name_logfile
    logger = logging.getLogger(name_logfile)
    formatter = logging.Formatter('[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s]: %(message)s')
    fileHandler = logging.FileHandler(path_logfile, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)

    logger.info(path_logfile)
    return logger


def read_file(fpath):
    with open(fpath, 'r') as f:
        res = json.load(f)
    return res


def load_data(fpath):
    convs = read_file(fpath)
    convertted_convs = []
    for conv in convs:
        sorted_conv = sorted(conv, key=lambda x: datetime.strptime(x['time'], '%Y-%m-%d-%H-%M-%S'))
        tmp = {"senders": [], "texts": []}
        for cs in sorted_conv:
            tmp["senders"].append(cs['sender'])
            tmp["texts"].append(cs['text'])
        convertted_convs.append(tmp)
    return convertted_convs


def remove_control_char(text: str, keep_tab=False) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != 'C' or (ch == '\t' and keep_tab))


def normalize(text: str) -> str:
    text = ftfy.fix_text(text)
    text = traditionalTosimplified(text)
    text = text.replace("\t", " ")
    text = remove_control_char(text)
    text = REPLACE_CONT_BLANKS_RE.sub(" ", text)
    return text.strip()


def de_reply_tag(s):
    REPLY_MENTION_REGEX = re.compile(r"回复 *@.*?: *")
    return REPLY_MENTION_REGEX.sub("", s)


def de_hashtag(s):
    HASHTAG_REGEX = re.compile(r"#.*?# *")
    return HASHTAG_REGEX.sub("", s)


def de_url(s):
    URL_REGEX = re.compile(
        # r"(?:^|(?<![A-Za-z0-9\/\.]))"
        r"(?:^|(?<![A-Za-z0-9\/]))"
        # protocol identifier
        # r"(?:(?:https?|ftp)://)"  <-- alt?
        r"(?:(?:https?:?\/\/|ftp:\/\/|www\d{0,3}\.))"
        # user:pass authentication
        r"(?:\S+(?::\S*)?@)?" r"(?:"
        # IP address exclusion
        # private & local networks
        r"(?!(?:10|127)(?:\.\d{1,3}){3})"
        r"(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})"
        r"(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})"
        # IP address dotted notation octets
        # excludes loopback network 0.0.0.0
        # excludes reserved space >= 224.0.0.0
        # excludes network & broadcast addresses
        # (first & last IP address of each class)
        r"(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])"
        r"(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}"
        r"(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))"
        r"|"
        # host name
        r"(?:(?:[a-z\\u00a1-\\uffff0-9]-?)*[a-z\\u00a1-\\uffff0-9]+)"
        # domain name
        # r"(?:\.(?:[a-z\\u00a1-\\uffff0-9]-?)*[a-z\\u00a1-\\uffff0-9]+)*"
        r"(?:\.(?:[a-z\\u00a1-\\uffff0-9]-?)*[a-z\\u00a1-\\uffff0-9]+)*"
        # TLD identifier
        r"(?:\.(?:[a-z\\u00a1-\\uffff]{2,}))" r"|" r"(?:(localhost))" r")"
        # port number
        r"(?::\d{2,5})?"
        # resource path
        r"(?:\/[^\)\]\}\s\u4e00-\u9fa5]*)?",
        # r"(?:$|(?![\w?!+&\/\)]))",
        # @jfilter: I removed the line above from the regex because I don't understand what it is used for, maybe it was useful?
        # But I made sure that it does not include ), ] and } in the URL.
        flags=re.UNICODE | re.IGNORECASE,
    )
    return URL_REGEX.sub("", s)


def de_weibo_url(s):
    WEIBO_URL_REGEX = re.compile(
        r"(?:(?:https?:?\/\/|ftp:\/\/|www\d{0,3}\.)t\.cn?(\/[a-zA-Z0-9]{0,8})?)|([网页链接|链接])"
    )
    return WEIBO_URL_REGEX.sub("", s)


def de_mention(s):
    AT_MENTION = re.compile(r"@[0-9a-zA-Z\u4E00-\u9FA5\-\_]+")
    return AT_MENTION.sub("", s)


def de_phone(s):
    PHONE_REGEX = re.compile(r"\D\d{11}\D")
    return PHONE_REGEX.sub("<PHONE>", s)


def de_QQ(s):
    QQ_REGEX = re.compile(r"[qQ]{2,}\d{5,12}\D")
    return QQ_REGEX.sub(" ", s)


def traditionalTosimplified(content):
    line = Converter("zh-hans").convert(content)
    return line


def contain_chinese(s):
    IS_CHINESE = re.compile(r"[\u4E00-\u9FA5]")
    if not re.findall(IS_CHINESE, s):
        return False
    unvalid_tokens = re.compile(r"[a-zA-Z0-9零一二三四五六七八九十壹贰叁肆伍陆柒捌玖拾佰仟]+")
    if len(unvalid_tokens.sub("", s)) <= 1:
        return False
    return True


def blacklist(s, cleaner):
    dirty_words = cleaner.processor.extract_keywords(s)
    if len(dirty_words) > 0 and dirty_words[0].strip():
        return True
    return False


def no_special_tokens(s):
    unvalid_tokens = set(('图片评论', '分享图片', '直播回放'))
    for ut in unvalid_tokens:
        if ut in str(s):
            return False
    return True


def check_if_blog_valid(texts, cleaner):
    flag = True
    for text in texts:
        text = str(text)
        if blacklist(text, cleaner):
            flag = False
        if '盖楼' in text:
            flag = False
        if not no_special_tokens(text):
            flag = False
    return flag


def de_advertisement(convs):
    # remove conversation sets if the length response over 20 and appear 2 times
    resp_dict = collections.defaultdict(set)
    for line in convs:
        if len(line['texts']) >= 3:
            continue
        for i in range(1, len(line['texts'])):
            resp_dict[str(line['texts'][i])].add(line['texts'][i - 1])
    ad_resp_dict = set()
    for k, v in resp_dict.items():
        if len(k.replace(" ", "")) > 20 and len(v) > 30:
            ad_resp_dict.add(k)
    logger.info([(k, [v])for k, v in resp_dict.items() if k in ad_resp_dict])
    return ad_resp_dict


def de_specific(s):
    DE_SPECIFIC = {"[图片]", "［图片］", "{ n楷体 s14}", "{ }", "{\\1c&H4080FF&}", "我擦", "\u200b", "查看动图", "评论配图"}
    for pattern in DE_SPECIFIC:
        s = s.replace(pattern, "")
    return s


def clean_text(s):
    s = clean(
        s,
        fix_unicode=False,
        to_ascii=False,
        normalize_whitespace=True,
        no_line_breaks=False,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        replace_with_url=" ",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>")
    return s


def process_utterance(s):
    s = normalize(str(s))

    #  remove 回复@xxx
    s = de_reply_tag(s)

    #  remove url
    s = de_url(s)

    #  remove weibo url
    s = de_weibo_url(s)

    #  remove hash tag
    s = de_hashtag(s)

    #  remove @
    s = de_mention(s)

    # remove QQ
    s = de_QQ(s)

    # repalce phone
    s = de_phone(s)

    # cleantext
    s = clean_text(s)

    # remove '查看动图' '评论配图'
    s = de_specific(s)

    s = REPLACE_CONT_BLANKS_RE.sub(" ", s)

    if not contain_chinese(s):
        s = ""
        return s

    return s.strip()


def ngrams(resp, n):
    ngram = list()
    if len(resp) >= n:
        for i in range(len(resp) - n + 1):
            ngram.append(resp[i: i + n])
    return ngram


def is_blandness(text, bland):
    tri_grams = ngrams(text, 3)
    cnt = collections.Counter(tri_grams)
    for word, num in cnt.items():
        if word in bland and 0.9 < (num * 3 / len(text)):
            return True
    return False


def b2i(b: bytes) -> int:
    return np.frombuffer(b[:HASH_SIZE], dtype=HASH_TYPE, count=1, offset=0).item(0)


def str_hash(s: str) -> int:
    h = hashlib.sha1(bytes(s, encoding="utf-8"))
    return b2i(h.digest())


def save_file(convs, filename):
    if convs:
        with open(filename, 'w') as f:
            json.dump(convs, f, ensure_ascii=False)


def statistics(convs):
    # 统计两人对话及两人以上的对话
    twopersons = 0
    multipersons = 0
    # 两人对话并且轮数大于1
    suitable_convs = 0
    for line in convs:
        if len(set(line['senders'])) == 2:
            twopersons += 1
            if len(line['senders']) > 2:
                suitable_convs += 1
        else:
            multipersons += 1
    return suitable_convs, twopersons, multipersons


def main_filter(fpath, out_dir, remove_blandness=False):
    log_path = os.path.join(os.path.dirname(out_dir), 'processed_rb_log/')
    global logger
    logger = setup_logger(os.path.basename(fpath) + 'processed.log', log_path)

    convs = load_data(fpath)
    advs = de_advertisement(convs)
    cleaner = Cleaner()
    bland = Blandness().load("/mnt/cfs/weibo_comments/blandness.json")
    ids = set()

    final = []
    for line in convs:
        if check_if_blog_valid(line['texts'], cleaner):
            processed = {'senders':[], 'texts':[]}
            for se, cv in zip(line["senders"], line["texts"]):
                try:
                    tp_cv = process_utterance(cv)
                    # and cv not in advs
                    if len(tp_cv) > 1 and cv not in advs:
                        if remove_blandness:
                            if is_blandness(tp_cv, bland):
                                if len(processed['senders']) >= 1:
                                    processed['senders'].pop(-1)
                                    rd = processed['texts'].pop(-1)
                                    # logger.info("Removed {} {}".format(rd, tp_cv))
                                    continue

                        processed['senders'].append(se)
                        processed['texts'].append(tp_cv)

                except:
                    logger.exception('Got exception on main handler')

            if processed['texts'] and len(processed['texts']) > 1:
                if len(processed['texts']) == 2 and processed['texts'][0] == processed['texts'][1]:
                    continue
                hash = str_hash(" ".join([t for t in processed['texts']]))
                if hash not in ids:
                    final.append(processed)
                    ids.add(hash)

    a, b, c = statistics(final)
    logger.info("多轮两人对话占比: {:.2f}，两人对话数目: {}, 多人对话数目: {}".format(a/b if b!= 0 else a, b, c))
    output = os.path.basename(fpath).split('.')[0] + '_processed.json'
    save_file(final, os.path.join(out_dir, output))


if __name__ == "__main__":
    base_dir = "/mnt/cfs/weibo_comments/formatted"
    main_filter(os.path.join(base_dir, sys.argv[1]), "/mnt/cfs/weibo_comments/processed", sys.argv[2])

