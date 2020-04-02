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
movie_dict = pickle.load(open("{}/movie_dict33.pkl".format(path), "rb"))


# total interacted movies by active users
movie_list=[]
for period in range(1,17):
    for user,item in active_user_dict.items():
        movie_list.append(item[period])
movie_list=list(set([l for sublist in movie_list for l in sublist]))

# movie doesn't have information
no_movie=[]
for movie in movie_list:
    try:
        mov=movie_dict[movie]
    except:
        no_movie.append(movie)

#movie latent representation
n_movie=17770
laten_factor=20
movie_dict={}
movie_matrix=torch.randn(n_movie,laten_factor,dtype=torch.float)
for i in movie_list:
    movie_dict[i]=movie_matrix[i-1]

#neural network
class simple_model(torch.nn.Module):
    def __init__(self,lf,hs):
        super(simple_model,self).__init__()
        self.fc1 = torch.nn.Linear(lf, hs)
        self.output_layer = torch.nn.Linear(hs, lf)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        u_rep = self.output_layer(x)
        return u_rep


hidden_size=64
rec_model = simple_model(laten_factor,hidden_size)

#total movies in each period
period_movies={}
for period in range(1,17):
    movies=[]
    for user,item in active_user_dict.items():
        movies.append(item[period])
    movies=list(set([l for sublist in movies for l in sublist]))
    period_movies[period]=movies


for user, item, labels in zip(active_user_dict.keys(),active_user_dict.values() ,active_label_dict.values()):

    for period, movie, label in zip(item.keys(),item.values(), labels.values()):
        movie_list_train = []
        movie_list_test = []
        test_indx = []

        #random split
        seed(1)
        for _ in range(0, 5):
            indx = randint(0, len(movie) - 1)
            test_indx.append(indx)
        indxes = [i for i in range(0, len(movie) - 1)]
        train_indx = list(set(indxes) - set(test_indx))
        train_movie = [movie[m] for m in train_indx]
        test_movie = [movie[m] for m in test_indx]
        train_label = [label[m] for m in train_indx]
        test_label = [label[m] for m in test_indx]

        train_tensor = []
        for mov in train_movie:
            movie_info = movie_dict[mov]
            train_tensor.append(movie_info.float())

        train_tensor = torch.mean(torch.stack(train_tensor),0)

        test_tensor = []
        for mov in test_movie:
            movie_info = movie_dict[mov]
            test_tensor.append(movie_info.float())

        test_tensor = torch.mean(torch.stack(test_tensor),0)

        train_label = torch.unsqueeze(torch.tensor(train_label).float(), 1)
        test_label = torch.unsqueeze(torch.tensor(test_label).float(), 1)

        total_items = []
        for mov in period_movies[period]:
            movie_info = movie_dict[mov]
            total_items.append(movie_info.float())

        item_t = torch.stack(total_items)

        #call neural network
        while True:
            r_u=rec_model(train_tensor)
            y_pred=torch.mm(item_t,r_u.T)
            print()

