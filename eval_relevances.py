
# return a list of relevances for supplied ids, given a dictionary of ratings
# unrated items get a score of zero
def get_relevances( ratings, ids ):
    # relevance score = rating - relevance_bias
    # e.g. 1...5 -> -2...2
    relevance_bias = 0 #changed 3.0 -> 5.5
    r = [ 0. ] * len( ids )
    for i, x in enumerate( ids ):
        try:
            if ratings[x] == -1:
                r[i] = 5.5
            else:
                r[i] = ratings[x] - relevance_bias
        except KeyError:
            pass

        return r


# returns unsorted best possible relevances from the test set
# if the list is shorter than ndcg_k, it gets padded with zeros
def get_best_relevances( ratings, ids, ndcg_k ):
    
    # relevance score = rating - relevance_bias
    # e.g. 1...5 -> -2...2
    relevance_bias = 0 #changed 3.0 -> 5.5
    r = [ ratings[x] - relevance_bias for x in ids if x in ratings and ratings[x] > relevance_bias ]

    if len( r ) < ndcg_k:
        return r + [ 0. ] * ( ndcg_k - len( r ))
    else:
        return r


def get_worst_relevances( ratings, ids, ndcg_k ):
    # relevance score = rating - relevance_bias
    # e.g. 1...5 -> -2...2
    relevance_bias = 5.5 #changed 3.0 -> 5.5
    r = [ ratings[x] - relevance_bias for x in ids if x in ratings and ratings[x] < relevance_bias ]
    if len( r ) < ndcg_k:
        return r + [ 0. ] * ( ndcg_k - len( r ))
    else:
        return r
