import requests
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import random
import multiprocessing
tqdm.pandas()


import requests
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

class LocalHostAdapter(HTTPAdapter):
    def __init__(self, source_address, *args, **kwargs):
        self.source_address = source_address
        super(LocalHostAdapter, self).__init__(*args, **kwargs)

    def init_poolmanager(self, *args, **kwargs):
        kwargs['source_address'] = (self.source_address, 0)  # Bind to the specified IP
        return super(LocalHostAdapter, self).init_poolmanager(*args, **kwargs)

def get_frequency(concept):
    try:
        url = "http://books.google.com/ngrams/json"

        query_params = {
            "content": concept,
            "year_start": 1800,
            "year_end": 2010,
            "smoothing": 1,
            "case_insensitive": True
        }
        # Create a session and mount the adapter to it
        
        # session = requests.Session()
        # session.mount('http://', LocalHostAdapter(rand_ip))

        # Make the request using the bound local IP
        response = requests.get(url=url, params=query_params)
        response = response.json()
        
        if len(response) == 0:
            return 0
        frequency = np.mean(response[0]['timeseries'])
        
        return frequency
    except:
        time.sleep(0.5)
        return get_frequency(concept)
        
if __name__ == "__main__":
    fpath = [
        "analysis/ccpt_bigram.csv",
        "analysis/conceptnet_bigram.csv",
    ]
    for path in fpath:
        df = pd.read_csv(path)
        df['root_freq'] = df['root'].progress_apply(get_frequency)
        df['modifier_freq'] = df['modifier'].progress_apply(get_frequency)
        df['co_freq'] = df['combination'].progress_apply(get_frequency)
        df.to_csv(path.replace(".csv", "_googlengram.csv"), index=False)