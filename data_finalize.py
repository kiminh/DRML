import pandas as pd
import torch
import pickle
import re
#Netflix dataset
path = "/home/krishna/Desktop/Dryu/dyRec/netflix-prize-data"
info_data_path = "{}/cooked_movies.csv".format(path)
movie_data = pd.read_csv(
            info_data_path,
            sep=",", engine='python')
print()