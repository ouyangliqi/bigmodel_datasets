import glob
import json
import os
import sys
import threading
import time
from multiprocessing import Pool

# import torch
from datasketch import MinHash, MinHashLSH


def query_minhash(fin, fout):
    mset = MinHashLSH(threshold=0.95, num_perm=100)

    orig_doc = 0
    res_doc = 0
    fout = open(fout, "w")
    with open(fin, 'r') as f:
        for line in f:
            doc = json.loads(line[:-1])
            orig_doc += 1

            l = MinHash(num_perm=100)  # a post

            words = set([w for doc_line in doc['cont'] for w in doc_line])

            for word in words:
                l.update(word.encode('utf-8'))
            # l.update("".join(words).encode('utf-8'))

            similar = len(mset.query(l))

            if similar == 0:
                mset.insert(line, l)

                res_doc += 1
                # doc['perplexity'] = batch_perplexity(doc['cont'])
                fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
        fout.close()
    return "orig_doc: {}, res_doc: {}, file: {}".format(orig_doc, res_doc, fin)


subdir = sys.argv[1]
# subdir = "/mnt/cfs/commoncrawl-2021-01-filter"
start = time.time()

# mset = MinHashLSH(threshold=0.95, num_perm=100)
# mset_lock = threading.Lock()

p = Pool(20)
jobs = []
for fn in glob.glob(subdir + "/**.txt"):
    if fn.startswith("under"):
        continue

    pos = fn.rfind(".")
    fn_out = fn[:pos] + "_dedup" + fn[pos:]
    fn_out = fn_out.replace(
            os.path.basename(os.path.dirname(fn)),
            # os.path.join(os.path.dirname(fn), 'minhash')  # target dir
            os.path.basename(os.path.dirname(fn))[:19] + "-s2-dedup"
        )
    # raise ValueError(fn_out)
    jobs.append(p.apply_async(query_minhash, (fn, fn_out)))

p.close()
p.join()

for job in jobs:
    print(job.get())

end = time.time()
print("time used: {}".format(end - start))













# from transformers import BertTokenizer, GPT2LMHeadModel

# tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
# model = GPT2LMHeadModel.from_pretrained("/mnt/cfs/models/uer/gpt2-chinese-cluecorpussmall")

# device = "cuda"
# model.cuda()


# def batch_perplexity(document):
#     encodings = tokenizer("\n\n".join(document), return_tensors="pt")

#     max_length = model.config.n_positions
#     stride = 512

#     nlls = []
#     for i in range(0, encodings.input_ids.size(1), stride):
#         begin_loc = max(i + stride - max_length, 0)
#         end_loc = min(i + stride, encodings.input_ids.size(1))
#         trg_len = end_loc - i  # may be different from stride on last loop
#         input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
#         target_ids = input_ids.clone()
#         target_ids[:, :-trg_len] = -100

#         with torch.no_grad():
#             outputs = model(input_ids, labels=target_ids)
#             neg_log_likelihood = outputs[0] * trg_len

#         nlls.append(neg_log_likelihood)

#     ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
#     return ppl.tolist()
