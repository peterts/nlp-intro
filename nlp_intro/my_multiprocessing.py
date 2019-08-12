import psutil
from pathos.multiprocessing import ProcessingPool
from time import sleep, time


MAX_WORKERS = psutil.cpu_count(logical=False)


def foo(text):
    sleep(0.1)
    return text.lower()


def process(f, iterable, n_workers=MAX_WORKERS):
    if n_workers < 0:
        n_workers = MAX_WORKERS + 1 + n_workers
    assert n_workers > 0, f"n_workers must be between {-MAX_WORKERS} and {MAX_WORKERS}"
    if n_workers == 1:
        return [f(x) for x in iterable]
    pool = ProcessingPool(nodes=4)
    return pool.map(f, iterable)

