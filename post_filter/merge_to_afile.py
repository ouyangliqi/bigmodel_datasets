import glob
import json
import os

commoncrawl = "/mnt/cfs/weibo_comments/processed"

fout = open("/mnt/cfs/weibo_comments/weibocomments_all.txt", "w")
nline = 0
for dir in glob.glob(commoncrawl):
    for file in glob.glob(dir + "/**"):
        with open(file) as fi:
            lines = json.load(fi)
            lines = lines[0] if len(lines) == 1 else lines
            if type(lines) != list:
                continue
            for conv in lines:
                fout.write(json.dumps({"texts": conv["texts"]}, ensure_ascii=False) + "\n")
                nline += 1
print(fout, nline)


# lccc = "/mnt/cfs/LCCC/processed"

# fout = open("/mnt/cfs/LCCC/lccc_all.txt", "w")
# nline = 0
# file = "/mnt/cfs/LCCC/LCCD.json"
# with open(file) as fi:
#     lines = json.load(fi)
#     for conv in lines:
#         fout.write(json.dumps({"texts": conv}, ensure_ascii=False) + "\n")
#         nline += 1
# print(file, nline)
