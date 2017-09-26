'''
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import heapq
from sklearn import cluster
from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction import DictVectorizer
from bottleneck import argpartition


# In[2]:


class Rec():
    
    def __init__(self):
        self.anime_tag_vector = pd.read_csv('anime_tag_vector.csv')
        self.anime_id = self.anime_tag_vector['anime_id'].as_matrix().tolist()
        self.anime_rating = self.anime_tag_vector['rating']
        del self.anime_tag_vector['anime_id']
        del self.anime_tag_vector['rating']
        
        # user instance: calculated rating to each tag
        self.instance = pd.read_csv('user_instances.csv')
        self.instance = self.instance.as_matrix()[1:]
        self.instance = [ row[1:] for row in self.instance ]
        
        # user actual rating to each watched anime
        self.rate = pd.read_csv('rating.csv')
        self.rate = self.rate.as_matrix().tolist()
        
        self.c = 50
        self.kmeans = cluster.KMeans(n_clusters=self.c, max_iter=300, init='k-means++',n_init=10, verbose=True, n_jobs=-1).fit(self.instance)
        self.clus = self.kmeans.predict(self.instance)
        
        # user watched list
        self.watch = [[] for _ in self.instance]
        for r in self.rate:
            self.watch[r[0]-1].append(r[1])

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
        
        self.anime_candidates = []
        for i in range(self.c):
            index = [user for user, x in enumerate(self.clus.tolist()) if x == i]
            self.anime_candidates.append(list(set([w for i in index for w in self.watch[i] if w in self.anime_id])))

        
    def get_ranked_ids(self, datadata, k):
        
        # generate user instance from data
        avg_rating = 0
        watch_times = 0
        data = dict(datadata)
        rating_times = 0
        overall_rating = 0
        total_rating_times = 0
        for r in data.values():
            watch_times += 1
            if r != -1:
                avg_rating += r
                rating_times += 1
                overall_rating += r
                total_rating_times += 1
        
        np.seterr(divide='ignore', invalid='ignore')
        # print(avg_rating)
        # avg_rating = np.nan_to_num(avg_rating / rating_times)
        
        # avg_overall_rating = overall_rating / total_rating_times
        avg_overall_rating = 7.8
        
        ratings = np.zeros(43)
        user_tags = np.zeros(43)
        # for anime_id, r in zip(data.keys(), data.values()):
            # if anime_id not in self.inverse_anime_id.keys():
                # continue
            # user_tags += self.genre[self.inverse_anime_id[anime_id]]
            # if rating_times == 0:
                # ratings += self.genre[self.inverse_anime_id[anime_id]] * 5
            
            # elif r == -1:
                # ratings += self.genre[self.inverse_anime_id[anime_id]] * (10 - avg_rating)
            # else:
                # ratings += self.genre[self.inverse_anime_id[anime_id]] * r
        
        # user_instance = np.nan_to_num(ratings / user_tags)
        user_instance = [0.5] * 43
        
        ins = user_instance
        
        watch = []
        for r in data.keys():
            watch.append(r)
        
        ins_watch = watch

        # predict!!!!
        pred = self.kmeans.predict([ins])[0]
        #index = [i for i, x in enumerate(self.clus.tolist()) if x == pred]
        #anime_candidates = [w for i in index for w in self.watch[i] if w in self.anime_id]
        #anime_candidates = set(anime_candidates) - set(ins_watch)

        k = min(k, len(self.anime_candidates))

        b = [self.anime_id.index(c) for c in self.anime_candidates[pred]]
        r = self.anime_rating[b].as_matrix()
        p = argpartition(r, len(self.anime_candidates) - k)
        return [self.anime_id[pp] for pp in p[len(self.anime_candidates) - k:]]
        
        #return heapq.nlargest(k,self.anime_candidates[pred], key=lambda x: self.anime_rating[self.anime_id.index(x)])

'''

# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import heapq
from sklearn import cluster
from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction import DictVectorizer
from bottleneck import argpartition

from math import ceil
from sklearn.neighbors import LSHForest


# In[2]:


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
        self.c = 400
        self.percentage = 100
        self.kmeans = cluster.KMeans(n_clusters=self.c, max_iter=300, init='k-means++',n_init=10, verbose=True, n_jobs=-1).fit(self.instance)
        self.clus = self.kmeans.predict(self.instance)
        
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
            
            users_in_clus = np.where(self.clus==i)[0]

            lshf = LSHForest(random_state=42)
            lshf.fit(self.instance[users_in_clus])  # X_train
            LSHForest(min_hash_match=4, n_candidates=50, n_estimators=10,
                          n_neighbors=5, radius=1.0, radius_cutoff_ratio=0.9,
                          random_state=42)
            num_representatives = ceil(len(users_in_clus)/ self.percentage)
            distances, indices = lshf.kneighbors([self.kmeans.cluster_centers_[i]], n_neighbors=num_representatives) #X_test
            representative = list(set([watch_anime for represent in indices for watch_list in np.array(self.watch)[represent] \
                                       for watch_anime in watch_list if watch_anime in self.anime_id]))
            self.anime_candidates.append(sorted(representative, key=lambda x: self.anime_rating[self.anime_id.index(x)], reverse=True))

            
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
        
        #return list(self.anime_candidates[pred]-set(ins_watch))[:k]

        # b = [self.anime_id.index(c) for c in self.anime_candidates[pred]]
        # r = self.anime_rating[b].as_matrix()
        # p = argpartition(r, len(self.anime_candidates) - k)
        # return [self.anime_id[pp] for pp in p[len(self.anime_candidates) - k:]]
        
        #return heapq.nlargest(k,self.anime_candidates[pred], key=lambda x: self.anime_rating[self.anime_id.index(x)])

