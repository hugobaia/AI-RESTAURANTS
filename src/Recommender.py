
################# IMPORTS #################

# Import Pandas
import pandas as pd

#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

import json
import functools
import Util

############################################

index = -1

# Function to calculate the price proximity with the desired price
def price_proximity(desiredPrice, restaurant):
    price = restaurant['price_avg']
    return abs(price- int(desiredPrice))

# Function that computes the weighted rating of each restaurant
def weighted_rating(minimum_votes, rating_mean, users, maxPrice, sim_scores, restaurant):
    vote_count = restaurant['vote_count']
    rating = restaurant['rating']
    price = restaurant['price_proximity']
    global index

    price_normalized = 0
    if (maxPrice != 0):
        price_normalized = 1 - (float(price) / float(maxPrice))

    price_normalized = price_normalized * 0.25
    votes = (((vote_count/(vote_count+minimum_votes) * rating) + (minimum_votes/(minimum_votes+vote_count) * rating_mean)) / 5) * 0.25
    similarity = (sim_scores[index][1] * 0.25)
    location = Util.calculateDistanceScore(users, restaurant) * 0.25

    index = index+1

    return (votes + similarity + location + price_normalized)

def getRestaurants(tags, users, desiredPrice):

    global index
    index = 0

    # Load Movies Metadata
    metadata = pd.read_csv('data/restaurants.csv', low_memory=False)

    #Define a TF-IDF Vectorizer Object.
    tfidf = TfidfVectorizer()

    #Replace NaN with an empty string
    metadata['expertise'] = metadata['expertise'].fillna('')

    #Creating a sample restaurant with the tags we want
    metadata.loc[metadata.index.max() + 1] = tags
    
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

    # Return the most similar retaurants
    q_restaurants = metadata.iloc[rest_indices]

    # Calculate C
    rating_mean = q_restaurants['rating'].mean()

    # Calculate the minimum number of votes required to be in the chart
    minimum_votes = q_restaurants['vote_count'].quantile(0.2)

    # Filter out all qualified restaurants into a new DataFrame
    q_restaurants = q_restaurants.copy().loc[metadata['vote_count'] >= minimum_votes]

    # Apply function to calculate the price proximity with the desired price
    calculate_price_proximity = functools.partial(price_proximity, desiredPrice)
    q_restaurants['price_proximity'] = q_restaurants.apply(calculate_price_proximity, axis=1)
    
    # Get max value of price_proximity column
    maxPrice = q_restaurants['price_proximity'].max()

    # Define a new feature 'score' and calculate its value with `weighted_rating()`
    calculate_score = functools.partial(weighted_rating, minimum_votes, rating_mean, users, maxPrice, sim_scores)
    q_restaurants['score'] = q_restaurants.apply(calculate_score, axis=1)

    #Sort restaurants based on score calculated above
    q_restaurants = q_restaurants.sort_values('score', ascending=False)

    #Print the top 5 restaurants
    return q_restaurants[['name', 'score']].head(5).to_json();