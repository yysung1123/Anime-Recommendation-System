
# coding: utf-8

# In[78]:


import pandas as pd
import numpy as np
import heapq
from sklearn import cluster
from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction import DictVectorizer
from bottleneck import argpartition, partition
from math import ceil, isnan


# In[79]:


class Rec():

    def __init__(self):
        self.anime_tag_vector = pd.read_csv('anime_tag_vector.csv')
        self.anime_id = self.anime_tag_vector['anime_id'].as_matrix().tolist()
        self.anime_rating = self.anime_tag_vector['rating']
        del self.anime_tag_vector['anime_id']
        del self.anime_tag_vector['rating']
        self.anime_tag_vector = self.anime_tag_vector.as_matrix()

        # user instance: calculated rating to each tag
        self.instance = pd.read_csv('user_instances.csv')
        del self.instance['user_id']
        self.instance = self.instance[1:].as_matrix()


        # user actual rating to each watched anime
        self.rate = pd.read_csv('rating.csv')
        self.rate = self.rate.as_matrix().tolist()

        # cluster user to c groups
        self.c = 50
        self.kmeans = cluster.KMeans(n_clusters=self.c, max_iter=300, init='k-means++',n_init=10, verbose=True, n_jobs=-1).fit(self.anime_tag_vector)
        self.clus = self.kmeans.predict(self.anime_tag_vector)

        # user watched anime list
        self.watch = np.array([[] for _ in self.instance]).tolist() ## TODO
        for r in self.rate:
            self.watch[r[0]-1].append(r[1])

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

        # anime_candidates for each cluster (sorted)
        self.anime_candidates = []
        for i in range(self.c):

            print("Cluster ", i)
            animes_index_in_clus = np.where(self.clus==i)[0]
            animes_index_in_clus = [i for i in animes_index_in_clus if not isnan(self.anime_rating[i])]
            #self.anime_candidates.append(sorted(animes_in_clus, key=lambda x: self.anime_rating[self.anime_id.index(x)], reverse=True))
            self.anime_candidates.append([self.anime_id[anime_index] for anime_index in sorted(animes_index_in_clus, key=lambda x: self.anime_rating[x], reverse=True)])

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

        # generate user instance from data
        data = dict(datadata)

        rating_times = 0
        rating_appeartime = np.zeros(43)
        ratings = np.zeros(43)
        avg_rate = self.avg_rating(datadata)

        for obj in datadata:

            anime_id = obj[0]
            r = obj[1]

            if r == -1:
                ratings += self.anime_tag_vector[self.anime_id_to_idx[anime_id]] * (10 - avg_rate)
            else:
                ratings += self.anime_tag_vector[self.anime_id_to_idx[anime_id]] * r

            rating_appeartime += self.anime_tag_vector[self.anime_id_to_idx[anime_id]]


        np.seterr(divide='ignore', invalid='ignore')
        #ratings = np.nan_to_num(ratings / rating_appeartime)
        ins = np.nan_to_num(ratings / rating_appeartime)




        ins_watch = []
        for r in data.keys():
            ins_watch.append(r)
        ins_watch = set(ins_watch)
        #print(ins_watch)


        # predict!!!!
        pred = self.kmeans.predict([ins])[0]
        #index = [i for i, x in enumerate(self.clus.tolist()) if x == pred]
        #anime_candidates = [w for i in index for w in self.watch[i] if w in self.anime_id]
        #anime_candidates = set(anime_candidates) - set(ins_watch)

        #k = min(k, len(self.anime_candidates))

        #return heapq.nlargest(k,anime_candidates[pred], key=lambda x: self.anime_rating[self.anime_id.index(x)])


        ans = []
        count = 0
        for cand in self.anime_candidates[pred]:
            if cand in ins_watch:
                continue
            if count == k:
                break
            ans.append(cand)
            count += 1

        return ans
