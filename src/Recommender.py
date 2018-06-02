
################# IMPORTS #################

# Import Pandas
import pandas as pd

#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

import json

from User import User

import Util

############################################

index = -1

def getRestaurants(tags, users, usersPrice):

    global index
    index = 0

    # Load Movies Metadata
    metadata = pd.read_csv('data/restaurants.csv', low_memory=False)

    #Define a TF-IDF Vectorizer Object.
    tfidf = TfidfVectorizer()

    #Replace NaN with an empty string
    metadata['expertise'] = metadata['expertise'].fillna('')
    
    def creatingNewRestaurant(tags):
        metadata.loc[metadata.index.max() + 1] = tags

    #Creating a sample restaurant with the tags we want
    creatingNewRestaurant(tags)
    
    #Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(metadata['expertise'])

    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    #Construct a reverse map of indices and restaurant names
    filt_rests = metadata.drop_duplicates('expertise')
    indices = pd.Series(filt_rests.index, index=filt_rests['expertise']).drop_duplicates()
    
    # Get the pairwsie similarity scores of all restaurants with that restaurant
    sim_scores = list(enumerate(cosine_sim[metadata.index.max()]))
    
    # Sort the restaurants based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 30 most similar restaurants
    sim_scores = sim_scores[1:31]

    # Get the restaurant indices
    rest_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar retaurants
    q_restaurants = metadata.iloc[rest_indices]

    # Calculate C
    C = q_restaurants['rating'].mean()

    # Calculate the minimum number of votes required to be in the chart, m
    m = q_restaurants['vote_count'].quantile(0.2)

    # Filter out all qualified restaurants into a new DataFrame
    q_restaurants = q_restaurants.copy().loc[metadata['vote_count'] >= m]

    def price_proximity(x):
        price = x['price_avg']
        return abs(price- int(usersPrice))

    q_restaurants['price_proximity'] = q_restaurants.apply(price_proximity, axis=1)
    
    maxPrice = q_restaurants['price_proximity'].max()

    # Function that computes the weighted rating of each restaurant
    def weighted_rating(x, m=m, C=C, users=users, maxPrice=maxPrice):
        v = x['vote_count']
        R = x['rating']
        price = x['price_proximity']
        global index

        price_normalized = 0
        if (maxPrice != 0):
            price_normalized = 1 - (float(price) / float(maxPrice))

        price_normalized = price_normalized * 0.3
        votes = (((v/(v+m) * R) + (m/(m+v) * C)) / 5) * 0.1
        similarity = (sim_scores[index][1] * 0.4)
        location = Util.calculateDistanceScore(users, x) * 0.2

        index = index+1

        return (votes + similarity + location + price_normalized)

    # Define a new feature 'score' and calculate its value with `weighted_rating()`
    q_restaurants['score'] = q_restaurants.apply(weighted_rating, axis=1)

    #Sort restaurants based on score calculated above
    q_restaurants = q_restaurants.sort_values('score', ascending=False)

    #Print the top 5 restaurants
    return q_restaurants[['name', 'score']].head(5).to_json();