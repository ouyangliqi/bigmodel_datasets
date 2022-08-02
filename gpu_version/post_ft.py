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
    NUM_GPUS = 4
    PROC_PER_GPU = 6
    queue = Manager().Queue()

    base_dir = "/mnt/cfs/"

    subdir = sys.argv[1]

    start = time.time()

    # initialize the queue with the GPU ids
    for gpu_ids in range(NUM_GPUS):
        for _ in range(PROC_PER_GPU):
            queue.put(gpu_ids)

    # p = Pool(NUM_GPUS * PROC_PER_GPU, maxtasksperchild=1000)

    p = Pool(20)
    for f in glob.glob(base_dir + subdir + "/*txt"):
        # if os.path.basename(f) in processed:
        #     print(f)
        #     continue
        p.apply_async(main_filter, args=(f, base_dir + subdir[0:19] + "-s3-filter", queue))

    p.close()
    p.join()
    p.terminate()


    end = time.time()
    print("time used: {}".format(end - start))


if __name__ == '__main__':
    main()
