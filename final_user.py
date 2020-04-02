import pandas as pd
import pickle
import numpy as np
import torch
from torch.nn import functional as F

from random import seed
from random import randint

path = "/home/krishna/Desktop/Dryu/dyRec/netflix-prize-data"
active_user_dict = pickle.load(open("{}/active_user_dict.pkl".format(path), "rb"))
active_label_dict = pickle.load(open("{}/active_user_label_dict.pkl".format(path), "rb"))
movie_dict = pickle.load(open("{}/movie_dict.pkl".format(path), "rb"))
user_list=[]
for user,item in active_user_dict.items():
    movie_list=[]
    flag=0
    for period in range(1,17):
        movie_list.append(item[period])
    movie_list=list(set([l for sublist in movie_list for l in sublist]))

    for movie in movie_list:
        try:
            mov=movie_dict[movie]
        except:
          flag=1
          break
    if flag==0:
        user_list.append(user)
final_user={}
final_label={}
for u in user_list:
    final_user[u]=active_user_dict[u]
    final_label[u]=active_label_dict[u]
pickle.dump(final_user, open("{}/final_user_dict.pkl".format(path), "wb"))
pickle.dump(final_label, open("{}/final_label_dict.pkl".format(path), "wb"))
print()

