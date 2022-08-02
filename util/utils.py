import collections
import glob
import json
import os
import pickle as pkl
import random
from cProfile import label

import fasttext
import numpy as np
import sentencepiece as spm
import torch
from flashtext import KeywordProcessor
from line_profiler import LineProfiler
from tqdm import tqdm
from zmq import device

from .data.textcnn.models.TextCNN import *

fasttext.FastText.eprint = lambda x: None

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


class Cleaner:
    def __init__(self, load_path=None) -> None:
        if load_path is not None:
            load_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "blacklist"), "dicty_words.dict")
            self.processor = self.load_words_processor(load_path)

    def load_words_processor(self, file_path):
        words_processor = KeywordProcessor()
        with open(file_path, "r", encoding="UTF-8") as f:
            for line in f:
                if len(line[:-1]) > 0 and len(line.strip()) > 0:
                    words_processor.add_keyword(line[:-1])
        return words_processor

    def load_black_list(self, dir):
        words_processor = KeywordProcessor()
        for file in glob.glob(dir):
            if 'gbk' in file:
                with open(file, "r", encoding="gbk") as f:
                    for line in f:
                        if len(line[:-1]) > 0 and len(line.strip()) > 1:
                            words_processor.add_keyword(line[:-1])
            else:
                with open(file, "r", encoding="UTF-8") as f:
                    for line in f:
                        if len(line[:-1]) > 0 and len(line.strip()) > 1:
                            words_processor.add_keyword(line[:-1])
        self.processor = words_processor


class Blandness:
    def __init__(self) -> None:
        pass

    def ngrams(self, resp, n):
        ngram = list()
        if len(resp) >= n:
            for i in range(len(resp) - n + 1):
                ngram.append(resp[i: i + n])
        return ngram

    def build(self, input_path, out_path):
        generic = collections.Counter()
        for file in tqdm(glob.glob(input_path)):
            with open(file, 'r') as f:
                res = json.load(f)
                for line in res:
                    for seq in line['texts']:
                        tri_grams = self.ngrams(seq, 3)
                        generic.update(list(set(tri_grams)))
        generic = generic = sorted(generic.items(), key=lambda x: x[1], reverse=True)
        generic = set(x for x, cnt in generic if cnt > 1000)

        with open(out_path, 'w', encoding='UTF-8') as f:
            f.write("\n".join(json.dumps(line, ensure_ascii=False) for line in generic))

    def load(self, data_path):
        with open(data_path, 'r', encoding='UTF_8') as f:
            return [json.loads(line) for line in f.readlines() if len(line.strip()) > 0]


def load_vocab(fn: str):
    res = []
    with open(fn, "r") as f:
        idx = 0
        for line in f:
            if idx == 0:
                idx += 1
                continue
            res.append(line.split(" ")[0])
    return res


class TextAuditing:
    def __init__(self) -> None:
        self.model = fasttext.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/fastxt_v3/text-auditing-model.ftz"))
        self.spm_model = spm.SentencePieceProcessor()
        self.spm_model.Load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/fastxt_v3/lang.model"))

    def predict(self, text):
        pre_text = " ".join(self.spm_model.EncodeAsPieces(text.strip()))
        label, pro = self.model.predict(pre_text)
        return label[0], pro[0]

    def batch_predict(self, texts):
        texts = [" ".join(self.spm_model.EncodeAsPieces(i.strip()[:256])) for i in texts]
        label, pro = self.model.predict(texts)
        labels = [i[0] for i in label]
        pros = [i[0] for i in pro]
        return labels, pros

class TextCNN:
    def __init__(self) -> None:
        self.embedding = 'embedding_SougouNews.npz'
        self.model_name = 'TextCNN'
        self.config = Config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/textcnn/textData"), self.embedding)

        self.tokenizer = lambda x: [y for y in x]

        if os.path.exists(self.config.vocab_path):
            self.vocab = pkl.load(open(self.config.vocab_path, 'rb'))
        else:
            self.vocab = self.build_vocab(self.config.train_path, tokenizer=self.tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
            pkl.dump(self.vocab, open(self.config.vocab_path, 'wb'))

        self.config.n_vocab = len(self.vocab)

        self.model = Model(self.config).to(self.config.device)

        self.model.load_state_dict(torch.load(self.config.save_path))

    def build_vocab(self, file_path, tokenizer, max_size, min_freq):
        vocab_dic = {}
        with open(file_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content = lin.split('\t')[0]
                for word in tokenizer(content):
                    vocab_dic[word] = vocab_dic.get(word, 0) + 1
            vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
            vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
            vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
        return vocab_dic

    def tokenize(self, text):
        tokens = [self.vocab.get(word, self.vocab.get(UNK)) for word in self.tokenizer(text)]
        pad_size = self.config.pad_size
        pad_id = self.vocab.get(PAD)
        if pad_size:
            if len(tokens) < pad_size:
                tokens.extend([pad_id] * (pad_size - len(tokens)))
            else:
                tokens = tokens[:pad_size]
        return tokens

    def predict(self, text):
        self.model.eval()
        text_token = torch.LongTensor([self.tokenize(text)]).to(self.config.device)
        with torch.no_grad():
            outputs = self.model(text_token)
            values, indices = torch.max(outputs, dim=1)

        return indices.item(), values.item()

    def batch_predict(self, texts):
        self.model.eval()
        labels, pros = [], []
        tokenized = [self.tokenize(text) for text in texts]
        text_token = torch.LongTensor(tokenized).to(self.config.device)
        with torch.no_grad():
            outputs = self.model(text_token)
            pro, label = torch.max(outputs, dim=1)
        labels = label.tolist()
        pros = pro.tolist()
        del text_token
        return labels, pros


if __name__ == "__main__":
    # blandness = Blandness()
    # blandness.build("/mnt/cfs/weibo_comments/processed/**", "/mnt/cfs/weibo_comments/blandness.json")

    # cleaner = Cleaner()
    # npc = json.load(open("npc_cdn.json"))
    # new_npc = {}
    # for k, v in npc.items():
    #     if len(cleaner.processor.extract_keywords(v)) > 0:
    #         print(cleaner.processor.extract_keywords(v))
    #         continue
    #     new_npc[k] = v
    # json.dump(new_npc, open("npc_cdn_new.json", "w"), ensure_ascii=False, indent=4)

    lprofiler = LineProfiler(TextAuditing.batch_predict, TextCNN.batch_predict, TextCNN.tokenize)

    texts = ["服务项目: 我们拥有坚持专业经营的理念『以客为尊』，提供客户最安心、最贴心、最用心的服务，并且提供全方位不动产经纪贴心服务。", "物件标的： 大湖国小;湖内国中附近 建物楼层： 共 3楼", "请告知房仲服务人员，您是在『104报纸房屋网 - 台庆冈山国宅店 』看到此物件，感恩您。", "(可择一输入即可, 请勿输入数字以外的任何符号, 输入时请连续输入数字即可。)", "您好~我想了解这个物件,请跟我联络,谢谢!", "点选填妥送出后,请稍后一段时间,静待系统将您的留言讯息发送简讯给该联络人。"]
    print(len(texts), len("".join(texts)))

    auditing = TextAuditing()
    cnn = TextCNN()

    lprofiler.enable()
    print(auditing.batch_predict(texts))

    print(cnn.batch_predict(texts))

    lprofiler.disable()
    lprofiler.print_stats()


