import re
import os
import json
import torch
import numpy as np
import random
import pickle
from tqdm import tqdm

from dataset import movielens_1m


def item_converting(row, rate_list, genre_list, director_list, actor_list):
    rate_idx = torch.tensor([[rate_list.index(str(row['rate']))]]).long()
    genre_idx = torch.zeros(1, 25).long()
    for genre in str(row['genre']).split(", "):
        idx = genre_list.index(genre)
        genre_idx[0, idx] = 1
    director_idx = torch.zeros(1, 2186).long()
    for director in str(row['director']).split(", "):
        idx = director_list.index(re.sub(r'\([^()]*\)', '', director))
        director_idx[0, idx] = 1
    actor_idx = torch.zeros(1, 8030).long()
    for actor in str(row['actors']).split(", "):
        idx = actor_list.index(actor)
        actor_idx[0, idx] = 1
    return torch.cat((rate_idx, genre_idx, director_idx, actor_idx), 1)


def user_converting(row, gender_list, age_list, occupation_list, zipcode_list): 
    gender_idx = torch.tensor([[gender_list.index(str(row['gender']))]]).long()
    age_idx = torch.tensor([[age_list.index(str(row['age']))]]).long()
    occupation_idx = torch.tensor([[occupation_list.index(str(row['occupation_code']))]]).long()
    zip_idx = torch.tensor([[zipcode_list.index(str(row['zip'])[:5])]]).long()
    return torch.cat((gender_idx, age_idx, occupation_idx, zip_idx), 1)


def load_list(fname):
    list_ = []
    with open(fname, encoding="utf-8") as f:
        for line in f.readlines():
            list_.append(line.strip())
    return list_


def generate(master_path):
    dataset_path = "movielens/ml-1m"
    rate_list = load_list("{}/m_rate.txt".format(dataset_path))
    genre_list = load_list("{}/m_genre.txt".format(dataset_path))
    actor_list = load_list("{}/m_actor.txt".format(dataset_path))
    director_list = load_list("{}/m_director.txt".format(dataset_path))
    gender_list = load_list("{}/m_gender.txt".format(dataset_path))
    age_list = load_list("{}/m_age.txt".format(dataset_path))
    occupation_list = load_list("{}/m_occupation.txt".format(dataset_path))
    zipcode_list = load_list("{}/m_zipcode.txt".format(dataset_path))

    dataset = movielens_1m()

    # hashmap for item information
    if not os.path.exists("{}/m_movie_dict.pkl".format(master_path)):
        movie_dict = {}
        for idx, row in dataset.item_data.iterrows():
            m_info = item_converting(row, rate_list, genre_list, director_list, actor_list)
            movie_dict[row['movie_id']] = m_info
        pickle.dump(movie_dict, open("{}/m_movie_dict.pkl".format(master_path), "wb"))
    else:
        movie_dict = pickle.load(open("{}/m_movie_dict.pkl".format(master_path), "rb"))
    # hashmap for user profile
    if not os.path.exists("{}/m_user_dict.pkl".format(master_path)):
        user_dict = {}
        for idx, row in dataset.user_data.iterrows():
            u_info = user_converting(row, gender_list, age_list, occupation_list, zipcode_list)
            user_dict[row['user_id']] = u_info
        pickle.dump(user_dict, open("{}/m_user_dict.pkl".format(master_path), "wb"))
    else:
        user_dict = pickle.load(open("{}/m_user_dict.pkl".format(master_path), "rb"))

