import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple
import math
import random

import pickle

from statsmodels.tsa.stattools import adfuller

import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

import torch
import torch.nn as nn
import torch.nn.functional as F

# depending on your OS
# --------------------
from tqdm import trange, tqdm
# from tqdm.notebook import trange,tqdm # works for Linux
# --------------------

from IPython.display import clear_output

import networkx as nx
import torch
from torch_geometric.data import InMemoryDataset, download_url

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.data import Data
from torch_geometric.nn import GATConv

from IPython.display import clear_output

from torch_geometric.loader import DataLoader

TYPE_OF_IDE = 'offline'  # ['colab', 'offline']

if TYPE_OF_IDE == 'colab':
    MDATA_PATH = '/content/drive/MyDrive/mdata'
    save_path = '/content/drive/MyDrive/GNNStorage/'

    # from google.colab import drive
    # drive.mount('/content/drive')

elif TYPE_OF_IDE == 'offline':
    MDATA_PATH = 'mdata/'
    save_path = 'saved/'

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def seed_everything(seed: int):
    r"""Sets the seed for generating random numbers in PyTorch, numpy and
    Python.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


random_seed = 42
seed_everything(random_seed)

def get_dates():
    return sorted(os.listdir(MDATA_PATH))


def get_tickers():
    dates = get_dates()
    tickers = []
    for date in dates:
        tickers += os.listdir(os.path.join(MDATA_PATH, date))
    return np.unique(tickers)


def get_market_data(ticker):
    agg_mmf_data_raw = []
    for date in get_dates():
        if not os.path.exists(os.path.join(MDATA_PATH, date, ticker)):
            continue
        df = pd.read_csv(
            os.path.join(MDATA_PATH, date, ticker),
            compression='gzip',
            dtype={'bar_time': float, 'TIMESTAMP': float},
            index_col=0,
            parse_dates=[2, 3],
            date_parser=pd.to_datetime,
        ).sort_values(by=['VOLUME', 'bar_count']).groupby('bar_time', as_index=False).last()
        agg_mmf_data_raw.append(df)
    agg_mmf_data_raw = pd.concat(
        agg_mmf_data_raw).set_index('bar_time').sort_index()

    agg_price_grid_raw = agg_mmf_data_raw.filter(like='PRICE_GRID')
    agg_vol_grid_raw = agg_mmf_data_raw.filter(like='VOL_GRID')

    agg_mmf_data_raw = agg_mmf_data_raw[agg_mmf_data_raw.columns[(
        ~agg_mmf_data_raw.columns.str.startswith('EXEC')
        &
        ~agg_mmf_data_raw.columns.str.startswith('PRICE_GRID')
        &
        ~agg_mmf_data_raw.columns.str.startswith('VOL_GRID')
    )]]
    agg_mmf_data_raw.drop([
        'TIMESTAMP', 'ZERO_SPREAD_ON_TRADE', 'EMPTY_LOB_SIDE_AFTER_UPDATE', 'NEGATIVE_SPREAD_AFTER_UPDATE',
        'ZERO_SPREAD_AFTER_UPDATE', 'EMPTY_LOB_SIDE_ON_TRADE', 'NEGATIVE_SPREAD_ON_TRADE', 'bar_count', 'WEEKDAY'
    ], inplace=True, axis=1)

    return agg_mmf_data_raw, agg_price_grid_raw


def get_adf_p_value(dtest):
    dftest = adfuller(dtest, maxlag=10)
    p_value = dftest[1]
    return p_value


def convert_list_of_dfs_to_one_multi_index_df(list_of_df, selected_tickers, resample=None):
  df = pd.concat(list_of_df, keys=selected_tickers)
  df.index.names = ['ticker', 'bar_time']
  df.index = df.index.set_levels(pd.to_datetime(df.index.levels[1]), level=1)

    # TODO : completely revise all resample procedure
  if resample is not None:
    df = df.groupby([pd.Grouper(level='ticker'),
                     pd.Grouper(level='bar_time', freq=resample)]
                    ).first()
  return df

print('DONE')