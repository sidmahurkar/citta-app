#IMPORTS
import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from nltk.cluster import KMeansClusterer
import nltk 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bert_embedding import BertEmbedding
from sklearn import cluster
from sklearn import metrics
from numpy import dot
from numpy.linalg import norm

bert_embedding = BertEmbedding()

#FUNCTIONS
def np_similarity(query_list, match_list):
    query_len = len(query_list)
    match_len = len(match_list)    
    common_count = len(list(set(query_list).intersection(set(match_list))))
    
    if ((query_len == 0) | (match_len == 0)):
    	return (-999)
    else:
    	return ( common_count / ((query_len * match_len)**0.5) )


def get_score(query, match_list, match_sentiment, alpha=0.95):
    #Preprocess the Query
    query = re.sub('\S*@\S*\s?', '', query)
    query = re.sub('\s+', ' ', query)
    query = re.sub("\'", "", query)
    
    #Extract NounPhrases and Sentiment
    b = TextBlob(query)
    query_list = b.noun_phrases
    sid = SentimentIntensityAnalyzer()
    query_sentiment = (sid.polarity_scores(query))['compound']
    
    #GetScore
    score = alpha * np_similarity(query_list, match_list) + (1-alpha) * abs(query_sentiment-match_sentiment)
    
    return score

def get_embedding(doc):
    sentences = doc.split('.')          
    embeddings = bert_embedding(sentences)
    
    mean_embedding = np.zeros(768)
    n_tokens = 0
    for sent in embeddings:
        n_tokens += len(sent[1])
        mean_embedding += np.sum(np.array(sent[1]), axis=0)       
    mean_embedding /= n_tokens
   
    return mean_embedding

def cosine_sim(a, b):
    return (dot(a, b)/(norm(a)*norm(b)))

def get_recommendation(query, df, df_topn, n=500, NUM_CLUSTERS=10):
    
    print("Got Top-N ....")
    
    #Get embedding for query
    query_embed = get_embedding(query)
    
    #Getting embedding dictionary
    text2embed = {}
    for i in range(len(df_topn)):
        text2embed.update({str(df_topn['wiki_id'][i]) : np.load('./embeddings/'+str(df['wiki_id'][i])+'.npy')})
        
    #Embedding array for clustering
    X = np.array(list(text2embed.values()))
    
    #Clustering
    kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
    kmeans.fit(X)
    labels = list(kmeans.labels_)
    centroids = kmeans.cluster_centers_
    print("Clustering done ....")
    
    #Find most semantically similar cluster head by cosine similarity
    cos_sim = []
    for centroid in centroids:
        cos_sim.append(cosine_sim(query_embed, centroid))
        
    max_index = cos_sim.index(max(cos_sim))    
    most_similar = centroids[max_index]
    most_sim_class = kmeans.predict(most_similar.reshape(1, -1))
    
    indices = [i for i, x in enumerate(labels) if x == most_sim_class]
    recommendations = []
    for i in indices:
        recommendations.append(df_topn.loc[i])
        
    return (recommendations)