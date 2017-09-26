
# coding: utf-8

# In[12]:


import numpy as np
from pandas import DataFrame
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KDTree


# In[14]:


class Rec():
    
    def __init__(self):
        
        ## tag_to_idx
        self.all_tags =["Action","Adventure","Cars","Comedy","Dementia","Demons","Drama","Ecchi","Fantasy","Game","Harem","Hentai","Historical","Horror","Josei","Kids","Magic","Martial Arts","Mecha","Military","Music","Mystery","Parody","Police","Psychological","Romance","Samurai","School","Sci-Fi","Seinen","Shoujo","Shoujo Ai","Shounen","Shounen Ai","Slice of Life","Space","Sports","Super Power","Supernatural","Thriller","Vampire","Yaoi","Yuri"]
        self.tag_to_idx = {}
        for i in range(0, 43):
            self.tag_to_idx.update({self.all_tags[i]:i})
        
        ## 0 - base
        anime_tag_vector_inputfile = pd.read_csv("anime_tag_vector.csv")
        self.anime_tag_vector = DataFrame(anime_tag_vector_inputfile, columns=self.all_tags).values
        self.anime_id = DataFrame(anime_tag_vector_inputfile, columns=["anime_id"]).values
        self.anime_id = self.anime_id.reshape(12294)
        self.anime_rating = DataFrame(anime_tag_vector_inputfile, columns=["rating"]).values
        self.anime_rating = self.anime_rating.reshape(12294)
        self.normalize_anime_tag_vector = preprocessing.normalize(self.anime_tag_vector)

        ## anime_id_to_idx
        self.anime_id_to_idx = {}
        for i in range(0, 12294):
            self.anime_id_to_idx.update({self.anime_id[i]:i})
        self.kdt = KDTree(self.anime_tag_vector)
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning) 
    
    def avg_rating(self, train):
        
        rating_times = 0
        
        ratings = 0
        
        for r in train:
            anime_id = r[0]
            rating = r[1]
            if rating != -1:
                ratings += rating
                rating_times += 1
                

        np.seterr(divide='ignore', invalid='ignore')        
        if rating_times == 0:
            ratings_vector = 5
        else:
            ratings_vector = np.nan_to_num(ratings / rating_times)
        
        return ratings_vector
        
    def get_ranked_ids(self, train, kin):
        
        rating_times = 0
        rating_appeartime = np.zeros(43)
        ratings = np.zeros(43)
        avg_rate = self.avg_rating(train)
        
        for obj in train:
            
            anime_id = obj[0]
            r = obj[1]
        
            if r == -1:
                ratings += self.anime_tag_vector[self.anime_id_to_idx[anime_id]] * (10 - avg_rate)
            else:
                ratings += self.anime_tag_vector[self.anime_id_to_idx[anime_id]] * r
                
            rating_appeartime += self.anime_tag_vector[self.anime_id_to_idx[anime_id]]
        
        
        np.seterr(divide='ignore', invalid='ignore')        
        ratings = np.nan_to_num(ratings / rating_appeartime)
    
        normalize_ratings = preprocessing.normalize(ratings)
        dist, idx = self.kdt.query(normalize_ratings, k = kin)
        idx = idx[0]
        
        OverRatingXAnime_id = []

        for i in idx:
            OverRatingXAnime_id.append((self.anime_rating[i], self.anime_id[i]))
        OD = sorted(OverRatingXAnime_id, key=lambda x: -x[0])
        
        ans = []
        for obj in OD:
            ans.append(obj[1])
        return ans
        
