import collections
import glob
import json
import os
import pickle as pkl

import fasttext
import numpy as np
import sentencepiece as spm
import torch
from flashtext import KeywordProcessor
from tqdm import tqdm

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
        self.model = fasttext.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/fastxt_v2/text-auditing-model.ftz"))
        self.spm_model = spm.SentencePieceProcessor()
        self.spm_model.Load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/fastxt_v2/lang.model"))


    def predict(self, text):
        pre_text = " ".join(self.spm_model.EncodeAsPieces(text.strip()))
        label, pro = self.model.predict(pre_text)
        return label[0], pro[0]

    def batch_predict(self, texts):
        texts = [" ".join(self.spm_model.EncodeAsPieces(i.strip()[:256])) for i in texts]
        label, pro = self.model.predict(texts)
        return label, pro


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


    def predict(self, text):
        words_line = []

        self.model.eval()
        token = self.tokenizer(text)
        seq_len = len(token)
        pad_size = self.config.pad_size
        if pad_size:
            if len(token) < pad_size:
                token.extend([PAD] * (pad_size - len(token)))
            else:
                token = token[:pad_size]
                seq_len = pad_size
        # word to id
        for word in token:
            words_line.append(self.vocab.get(word, self.vocab.get(UNK)))

        text_token = torch.LongTensor([words_line]).to(self.config.device)
        seq_len = torch.LongTensor([seq_len]).to(self.config.device)
        datas = (text_token, seq_len)
        with torch.no_grad():
            outputs = self.model(datas)
            values, indices = torch.max(outputs, dim=1)

        return indices.item(), values.item()

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

    auditing = TextAuditing()
    print(auditing.batch_predict(['顶尖波王005期：【顶尖波王】 模棱两可_六合宝典论坛', '下载【六合宝典】APP，随时随地看资料。', '首页', '走势', '更多', '资料', '投注', '香港挂牌', '曾道人论坛', '四不像图论坛', '红姐图库', '精准计划', '彩库宝典', '开奖记录', '开奖日期', '开奖数据', '全年图纸', '四不像论坛', '挂牌论坛', '看贴', '这料只为论坛忠实彩民（六合宝典）', '综合各大高手公式得出以下资料.值得参考', '005期:必中⑨肖:虎猴狗蛇兔马牛猪龙', '005期:必中⑥肖:虎猴狗蛇兔马', '005期:必中③肖:虎猴狗 必中③码:23.35.05', '005期:平特②肖:(虎拖猴) 平特②尾:(3尾拖5尾)', '2021提高网站速度,减少各位看官流量', '帖子不保留大量往期记录，2020的请自行保存！', '001期:↖↗顶尖波王══《蓝波防红波》══开:34准中', '002期:↖↗顶尖波王══《绿波防红波》══开:33准中', '003期:↖↗顶尖波王══《绿波防红波》══开:43准中', '004期:↖↗顶尖波王══《蓝波防红波》══开:02准中', '005期:↖↗顶尖波王══《蓝波防绿波》══开:00准中', '眼神韵律 投资稳赚【投资十码】已公开,中奖率99%', '人亦已歌 投资稳赚【平特投资】已公开,中奖率99%', '梦回旧景 理财稳赚【理财六肖】已公开,中奖率99%', '旧事惘然 计划稳赚【大小计划】已公开,中奖率99%', '上世笑眸 理财稳赚【赚钱家野】已公开,中奖率99%', 'Copyright ©2020 手机开奖站 0141000.com 请记好，以便再次访问 六合宝典论坛']))

