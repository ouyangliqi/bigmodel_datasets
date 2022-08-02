# -*- coding: utf8 -*-
import glob
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import Manager, Pool, current_process

from post_filter import *


def main():
    # processed = [i.strip() for i in open("processed.txt")]

    base_dir = "/mnt/cfs/"

    subdir = sys.argv[1]

    start = time.time()

    # p = Pool(NUM_GPUS * PROC_PER_GPU, maxtasksperchild=1000)
    with ProcessPoolExecutor(2) as executor:
        features = []
        for f in glob.glob(base_dir + subdir + "/*txt"):
            # if os.path.basename(f) in processed:
            #     print(f)
            #     continue
            features.append(executor.submit(main_filter, f, base_dir + subdir[0:19] + "-s3-filter", None))
        wait(features)

        # p.apply_async(main_filter, args=(f, base_dir + subdir[0:19] + "-s3-filter", queue))

    # p.close()
    # p.join()
    # p.terminate()


    end = time.time()
    print("time used: {}".format(end - start))


if __name__ == '__main__':
    main()
