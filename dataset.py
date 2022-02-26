
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from processing import *
import os 

class ESC50Data(Dataset):
  def __init__(self, base, df, in_col, out_col):
    self.df = df
    self.data = []
    self.labels = []
    self.c2i={}
    self.i2c={}
    self.categories = sorted(df[out_col].unique())
    for i, category in enumerate(self.categories):
      self.c2i[category]=i
      self.i2c[i]=category
    for ind in tqdm(range(len(df))):
      row = df.iloc[ind]
      file_path = os.path.join(base,row[in_col])
      self.data.append(spec_to_image(get_melspectrogram_db(file_path))[np.newaxis,...])
      self.labels.append(self.c2i[row['category']])
  def __len__(self):
    return len(self.data)
  def __getitem__(self, idx):
    return np.array(self.data[idx],dtype=np.float32), np.array(self.labels[idx])

def get_data():
    df = pd.read_csv('ESC-50-master/meta/esc50.csv')
    train = df[df['fold']!=5]
    valid = df[df['fold']==5]
    train_data = ESC50Data('ESC-50-master/audio', train, 'filename', 'category')
    valid_data = ESC50Data('ESC-50-master/audio', valid, 'filename', 'category')
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=16, shuffle=True)
    return train_loader, valid_loader
get_data()