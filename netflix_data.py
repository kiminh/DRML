import pandas as pd
import torch
import pickle
import re

#item embeddings
class item(torch.nn.Module):
    def __init__(self):
        super(item, self).__init__()
        self.num_rate = 10
        self.num_genre = 27
        self.num_director = 5078
        self.embedding_dim = 32

        self.embedding_rate = torch.nn.Embedding(
            num_embeddings=self.num_rate,
            embedding_dim=self.embedding_dim
        )

        self.embedding_genre = torch.nn.Linear(
            in_features=self.num_genre,
            out_features=self.embedding_dim,
            bias=False
        )

        self.embedding_director = torch.nn.Linear(
            in_features=self.num_director,
            out_features=self.embedding_dim,
            bias=False
        )


    def forward(self, rate_idx, genre_idx, director_idx):
        rate_emb = self.embedding_rate(rate_idx)[:, -1, :]
        genre_emb = self.embedding_genre(genre_idx.float()) / torch.sum(genre_idx.float(), 1).view(-1, 1)
        director_emb = self.embedding_director(director_idx.float()) / torch.sum(director_idx.float(), 1).view(-1, 1)
        return torch.cat((rate_emb, genre_emb, director_emb), 1)


#item conversion
def item_converting(row,director_list,genre_list):
    rate_idx = torch.tensor([[row['Rating']]]).long()
    genre_idx = torch.zeros(1, len(genre_list)).long()
    for genre in str(row['Genres']).split("-"):
        idx = genre_list.index(genre)
        genre_idx[0, idx] = 1
    director_idx = torch.zeros(1, len(director_list)).long()
    for director in str(row['Directors']).split("-"):
        idx = director_list.index(re.sub(r'\([^()]*\)', '', director))
        director_idx[0, idx] = 1
    return rate_idx, genre_idx, director_idx

#Netflix dataset
path = "/home/krishna/Desktop/Dryu/dyRec/netflix-prize-data"
info_data_path = "{}/cooked_movies.csv".format(path)
movie_data = pd.read_csv(
            info_data_path,
            sep=",", engine='python'
        ) #names=['id', 'year', 'title', 'Runtime','Rating','Directors','Writers','Production companies','Genres']
movie_data=movie_data.dropna()
director_list=movie_data.Directors.unique()
genre_list=movie_data.Genres.unique()

movie_director=[]
for director in director_list:
    if director.find('-')!=-1:
        temp=director.split('-')
        for direct in temp:
            movie_director.append(direct)
    else:
        movie_director.append(director)
movie_director=list(set(movie_director))

movie_genre=[]
for genre in genre_list:
    if genre.find('-')!=-1:
        temp=genre.split('-')
        for gen in temp:
            movie_genre.append(gen)
    else:
        movie_genre.append(genre)
movie_genre=list(set(movie_genre))

movie_dict={}
uniq_mov=movie_data.id.unique()
movie_data['Rating']=movie_data['Rating'].astype(int)
uniq_rating=movie_data.Rating.unique()
print('Total movies='+str(len(uniq_mov)))
print('Total Rating='+str(len(uniq_rating)))
print('Total genre='+str(len(movie_genre)))
print('Total director='+str(len(movie_director)))

item_embed=item()
for _,row in movie_data.iterrows():
    rate_idx, genre_idx, director_idx=item_converting(row,movie_director,movie_genre)
    movie_dict[row['id']] =item_embed(rate_idx, genre_idx, director_idx)
pickle.dump(movie_dict, open("{}/movie_dict.pkl".format(path), "wb"))
print('Item dictionary is pickled.')