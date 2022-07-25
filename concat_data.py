import glob
import json
import os


def generate_offsets(self, filename):
    offsets = []
    with open(filename) as f:
        offsets.append(f.tell())
        while f.readline():
            offsets.append(f.tell())
    return offsets

def read_lines_from_iterator(self, data_path, offsets, begin_line, num_lines):
    # read with chunk
    with open(data_path) as f:
        f.seek(offsets[begin_line])
        for _ in range(num_lines):
            yield f.readline()


commoncrawl = "/mnt/cfs/commoncrawl-202*-**-filter/perplexity"
for dir in glob.glob(commoncrawl):
    nline = 0
    fout = open(dir + "under_all-data.txt", "w")
    for file in glob.glob(dir + "/**"):
        with open(file) as fi:
            for line in fi:
                fout.write(line)
                nline += 1
    print(dir, nline)





# save lines meta information

