import glob
import json
import os
from remove_codes import remove_codes
# commoncrawl = "/mnt/cfs/commoncrawl-202*-**-filter/minhash"

# for dir in glob.glob(commoncrawl):
#     fout = os.path.join(os.path.dirname(dir), "under_all-data.txt")
#     f = open(fout, "w")
#     nline = 0
#     lines = []
#     for file in glob.glob(dir + "/**"):
#         if not file.startswith("result"):
#             continue
#         with open(file) as fi:
#             for line in fi:
#                 if 20000 >= len("".join(json.loads(line)["cont"])) >= 10:
#                     lines.append(line)
#                     nline += 1
#     lines = sorted(lines, key=lambda x: len("".join(json.loads(x)["cont"])), reverse=False)
#     for l in lines:
#         f.write(l)
#     print(fout, nline)


def remove_unvalid_codes(texts):
    new_texts = []
    for s in texts:
        new_s = remove_codes(s)
        if len(s) != new_s:
            print(s)
            print(new_s)
        new_texts.append(new_s)
    return new_texts


dir = "/mnt/cfs/commoncrawl-2021-01-filter/minhash"
fout = os.path.join(os.path.dirname(dir), "under_all-data.txt")
f = open(fout, "w")
nline = 0
lines = []
for file in glob.glob(dir + "/result_**"):
    with open(file) as fi:
        for line in fi:
            if 20000 >= len("".join(json.loads(line)["cont"])) >= 10:
                lines.append(line)
                nline += 1
lines = sorted(lines, key=lambda x: len("".join(json.loads(x)["cont"])), reverse=False)
for l in lines:
    f.write(l)
print(fout, nline)







# /mnt/cfs/commoncrawl-2021-04-filter/perplexity 6623694
# /mnt/cfs/commoncrawl-2021-09-filter/perplexity 4513543
# /mnt/cfs/commoncrawl-2021-03-filter/perplexity 5129419
# /mnt/cfs/commoncrawl-2021-08-filter/perplexity 3994815
# /mnt/cfs/commoncrawl-2021-12-filter/perplexity 2425393
# /mnt/cfs/commoncrawl-2021-05-filter/perplexity 4890795
# /mnt/cfs/commoncrawl-2021-10-filter/perplexity 5907312
# /mnt/cfs/commoncrawl-2021-01-filter/perplexity 5954626
# /mnt/cfs/commoncrawl-2021-06-filter/perplexity 4453197
# /mnt/cfs/commoncrawl-2022-01-filter/perplexity 5528121


