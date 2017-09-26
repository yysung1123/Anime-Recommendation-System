
# coding: utf-8

# In[56]:


import pandas as pd
import numpy as np
import heapq
import math
from sklearn import cluster
from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction import DictVectorizer
from bottleneck import argpartition, partition
from math import ceil

class Rec():

    def __init__(self):
        self.anime_tag_vector = pd.read_csv('anime_tag_vector.csv')
        self.anime_id = self.anime_tag_vector['anime_id'].as_matrix().tolist()
        self.anime_rating = self.anime_tag_vector['rating']
        self.anime_pair = [p for p in list(zip(self.anime_id, self.anime_rating)) if not math.isnan(p[1])]

        # tag
        self.animes = pd.read_csv("anime.csv")
        tags = []
        for genres in self.animes["genre"].fillna(""):
            for genre in str.split(genres, ", "):
                tags.append(genre)
        tags = sorted(list(set(tags))[1:]) # Remove ''
        v = DictVectorizer(sparse="False")
        genre = v.fit_transform(self.animes["genre"].fillna(", ".join(tags)).apply(lambda x: {i: 1 for i in str.split(x, ", ")}))
        self.genre = np.array(genre.todense())

        self.inverse_anime_id = {}
        for index, anime in enumerate(self.anime_id):
            self.inverse_anime_id[anime] = index

        # anime_candidates (sorted)
        # self.anime_candidates = sorted(self.anime_id, key=lambda x: self.anime_rating[self.anime_id.index(x)], reverse=True)
        self.anime_candidates = sorted(self.anime_pair, key=lambda x: x[1], reverse=True)

        #for i in range(self.c):

         #   print("Cluster ", i)
         #   animes_in_clus = np.where(self.clus==i)[0]
            #self.anime_candidates.append(sorted(animes_in_clus, key=lambda x: self.anime_rating[self.anime_id.index(x)], reverse=True))
         #   self.anime_candidates.append(sorted(animes_in_clus, key=lambda x: self.anime_rating[x], reverse=True))

            #self.anime_candidates.append(list(set(np.where(self.clus==i)[0])))


        self.anime_id_to_idx = {}
        for i in range(0, 12294):
            self.anime_id_to_idx.update({self.anime_id[i]:i})


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

    def get_ranked_ids(self, datadata, k):

        data = dict(datadata)


        ins_watch = []
        for r in data.keys():
            ins_watch.append(r)
        ins_watch = set(ins_watch)
        #print(ins_watch)


        # predict!!!!
        #index = [i for i, x in enumerate(self.clus.tolist()) if x == pred]
        #anime_candidates = [w for i in index for w in self.watch[i] if w in self.anime_id]
        #anime_candidates = set(anime_candidates) - set(ins_watch)

        #k = min(k, len(self.anime_candidates))

        #return heapq.nlargest(k,anime_candidates[pred], key=lambda x: self.anime_rating[self.anime_id.index(x)])


        ans = []
        count = 0
        for cand in self.anime_candidates:
            if cand[0] in ins_watch:
                continue
            if count == k:
                break
            ans.append(cand[0])
            count += 1

        return ans
