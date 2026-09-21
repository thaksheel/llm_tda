import torch
from datasets import load_from_disk
import pandas as pd 
import numpy as np 


filename = "./datasets/harmful-tuning"
harmful_ds = load_from_disk(filename) 

df_harmful = pd.read_csv("./datasets/harmful-tuning_test.csv")
print(df_harmful.head())
print(harmful_ds)
