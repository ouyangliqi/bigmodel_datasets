# -*- coding: utf8 -*-
import glob
import logging
import sys
import time
from multiprocessing import Pool

from weibo_filter import *

base_dir = "/mnt/cfs/weibo_comments/formatted/"
out_dir = "/mnt/cfs/weibo_comments/processed_rb"


start = time.time()

p = Pool(20)

for f in glob.glob(base_dir + "/*json"):
    logging.info("Process file" + f)
    p.apply_async(main_filter, args=(f, out_dir, True))

p.close()
p.join()

# single process debug
# subdir = sys.argv[1]
# main_filter(base_dir + subdir, out_dir)

end = time.time()
logging.info("time used: {}".format(end - start))

