# -*- coding: utf8 -*-
import glob
import sys
import time
from multiprocessing import Pool

from post_filter import *

base_dir = "/mnt/cfs/"

subdir = sys.argv[1]

start = time.time()

p = Pool(20)

for f in glob.glob(base_dir + subdir + "/*txt"):
    p.apply(main_filter, args=(f, base_dir + subdir[0:19] + "-filter"))

p.close()
p.join()

end = time.time()
print("time used: {}".format(end - start))

