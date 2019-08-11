import numpy as np
import pandas as pd
from nlp_intro.diskusjon_no_scraper import DATA_FILE


def draw_uniform_data(n_points, *ranges):
    return np.asarray([np.random.uniform(*r, n_points) for r in ranges]).T


def load_diskusjon_no_data(max_label_level=0):
    df = pd.read_json(DATA_FILE, lines=True, encoding="utf8")
    if max_label_level is not None:
        if max_label_level == -1:
            max_label_level = 20  # We assume no label as a level higher than this
        df["label"] = df["labels"].apply(lambda labels: " > ".join(labels[:max_label_level+1]))
        df = df.drop("labels", axis=1)
    return df



