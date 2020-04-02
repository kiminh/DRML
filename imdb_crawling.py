import requests
from bs4 import BeautifulSoup
from imdb import IMDb
import pickle
from bs4 import Tag,NavigableString

path = "/home/krishna/Desktop/Dryu/dyRec/netflix-prize-data"
movie_id={}
url='https://www.imdb.com/search/title/?sort=num_votes,desc&start=1&title_type=feature&year=1874,2005' #'https://www.imdb.com/search/title/?title_type=feature&year=1874-01-01,2005-12-31&sort=num_votes,desc&after=WzE4LCJ0dDE2OTE4MjYiLDg4NDAxXQ%3D%3D'
for i in range(1,4902):
    flag=0

    r = requests.get(url)
    bs = BeautifulSoup(r.text,'html.parser')
    for a in bs.find_all('a',href=True):
        a_href=str(a['href'])
        if a_href.startswith('/title'):
            flag1 = 0
            temp_id=a_href.split('/')[2]
            temp_id=int(temp_id[2:])
            try:
                already_present=movie_id[temp_id]
            except:

                try:
                    tag_type=a.contents[0]
                    # print(tag_type)
                    if tag_type != 'See full summary':
                        if isinstance(tag_type, NavigableString):
                            content = str(a.contents[0])
                            if len(content) > 1 and (content[0].isalpha() or content[0].isdigit()):
                                movie_id[temp_id] = content
                except:
                    nothing_do=1


        if a_href.startswith('/search'):
            if flag == 0:
                tag_type = a.contents[0]
                if isinstance(tag_type,NavigableString):
                    if len(str(a.contents[0]))>3:
                        if str(a.contents[0]).startswith('Next'):
                            url='https://www.imdb.com'+a['href']
                            flag=1

    print('{} page crawled for title id.'.format(i))
    print(url)


with open('{}/movie_id.pkl'.format(path), 'wb') as f:
   pickle.dump(movie_id, f)

# movie_id = pickle.load(open("{}/movie_id.pkl".format(path), "rb"))
# #create an instance of the IMDb class from imdbpy api
# ia = IMDb()
# movie = ia.get_movie('Inception')
# movie_dict={}
# i=0
# for id in movie_id:
#     movie = ia.get_movie('{}'.format(id))
#     temp_dict={}
#     temp_dict['genres']=movie['genres']
#     temp_dict['plot'] = movie['plot'][0].split('::')[0]
#     temp_dict['ratings'] = movie['rating']
#     directors=[director['name'] for director in movie['directors']]
#     temp_dict['directors']=directors
#     casts=[cast['name'] for cast in movie['cast']]
#     temp_dict['actors'] = casts[0:5]
#     movie_dict[movie['title']]=temp_dict
#     if i%10==0:
#         print('{} crawled for movie information.'.format(i+1))
#     i += 1
#
# pickle.dump(movie_dict, open("{}/movie_dict_info.pkl".format(path), "wb"))
# print('Movie dictionary is pickled.')
#
