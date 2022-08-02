import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor as Pool
import logging
from logging.handlers import QueueHandler, QueueListener
from typing import Text
sys.path.append(os.path.abspath(".."))
from util.utils import *


def worker_init(q):
    # all records from worker processes go to qh and then into q
    qh = QueueHandler(q)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(qh)


def logger_init(name_logfile, path_logfile):
    q = mp.Queue()
    # this is the handler for all log records
    path_logfile = path_logfile + name_logfile
    formatter = logging.Formatter('[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s]: %(message)s')
    fileHandler = logging.FileHandler(path_logfile, mode='w')
    fileHandler.setFormatter(formatter)

    # ql gets records from the queue and sends them to the handler
    ql = QueueListener(q, fileHandler)
    ql.start()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # add the handler to the logger so records from this process are handled
    logger.addHandler(fileHandler)

    return ql, q


def chunkify_file(filepath, num_chunks, skiplines=-1):
    """
    function to divide a large text file into num_chunks chunks and the chunks are line aligned

    Params :
        fname : path to the file to be chunked
        num_chunks : number of chunks
        skiplines : number of lines in the begining to skip, -1 means don't skip any lines
    Returns :
        start and end position of chunks in Bytes
    """
    chunks = []
    file_end = os.path.getsize(filepath)
    print(f'file size : {file_end}')
    size = file_end // num_chunks

    with open(filepath, "rb") as f:
        if(skiplines > 0):
            for i in range(skiplines):
                f.readline()

        chunk_end = f.tell()
        count = 0
        while True:
            chunk_start = chunk_end

            f.seek(f.tell() + size, os.SEEK_SET)
            f.readline()  # make this chunk line aligned
            chunk_end = f.tell()
            chunks.append((chunk_start, chunk_end - chunk_start, filepath))
            count += 1


            if chunk_end >= file_end:
                break

    assert len(chunks) == num_chunks

    return chunks


def parallel_apply_line_by_line_chunk(chunk_data):
    """
    function to apply a function to each line in a chunk

    Params :
        chunk_data : the data for this chunk
    Returns :
        list of the non-None results for this chunk
    """
    chunk_start, chunk_size, file_path, func_apply = chunk_data[:4]
    func_args = chunk_data[4:]

    print(f'start {chunk_start}')

    i = 0
    st = time.time()

    # textcnn = TextCNN()
    fasttext_model = TextAuditing()
    # func_args.append(fasttext_model)

    res = []
    length = []
    with open(file_path, "rb") as f:
        f.seek(chunk_start)

        while True:
            i += 1

            line = f.readline().decode(encoding='utf-8')

            if line == '':
                # the last chunk of file ends with ''
                break

            ret, l = func_apply(line, fasttext_model, *func_args)
            length.append(l)

            if(ret != None):
                res.append(ret)

            if i % 1_000_000 == 0:
                ed = time.time()
                print(ed - st, f.tell() - chunk_start, '/', chunk_size, (f.tell() - chunk_start) / chunk_size)
                st = ed

            if f.tell() - chunk_start >= chunk_size:
                break
    # del textcnn
    # torch.cuda.empty_cache()
    return res, length


def parallel_apply_line_by_line(input_file_path, num_procs, func_apply, func_args, loger_args, skiplines=0, fout=None, merge_func=None):
    """
    function to apply a supplied function line by line in parallel

    Params :
        input_file_path : path to input file
        num_procs : number of parallel processes to spawn, max used is num of available cores - 1
        func_apply : a function which expects a line and outputs None for lines we don't want processed
        func_args : arguments to function func_apply
        skiplines : number of top lines to skip while processing
        fout : do we want to output the processed lines to a file
        merge_func : merge function dealing with outputs of chunks
    Returns :
        merged output
    """
    num_parallel = num_procs
    print(input_file_path)
    print(f'num parallel: {num_procs}')

    jobs = chunkify_file(input_file_path, num_procs, skiplines)

    jobs = [list(x) + [func_apply] + func_args for x in jobs]

    print("Starting the parallel pool for {} jobs ".format(len(jobs)))

    # q_listener, q = logger_init(*loger_args)

    # pool = mp.Pool(num_parallel, worker_init, [q], maxtasksperchild=1000)  # maxtaskperchild - if not supplied some weird happend and memory blows as the processes keep on lingering

    outputs = []
    length = []

    t1 = time.time()
    with Pool(num_parallel) as pool:
        chunk_outputs = pool.map(parallel_apply_line_by_line_chunk, jobs)
    # chunk_outputs = pool.map(parallel_apply_line_by_line_chunk, jobs)

    for i, output in enumerate(chunk_outputs):
        outputs.extend(output[0])
        length.extend(output[1])

    # pool.close()
    # pool.terminate()
    # q_listener.stop()

    if merge_func is not None:
        print('merging outputs...')
        output = merge_func(outputs)
    else:
        output = outputs

    print("All Done in time ", time.time() - t1)

    return output, sum(length)
