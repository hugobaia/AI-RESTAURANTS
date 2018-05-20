# Import Pandas
import pandas as pd

#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Load Movies Metadata
metadata = pd.read_csv('data/restaurants.csv', low_memory=False)

#Define a TF-IDF Vectorizer Object.
tfidf = TfidfVectorizer()

#Replace NaN with an empty string
metadata['expertise'] = metadata['expertise'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(metadata['expertise'])

#Output the shape of tfidf_matrix
print(tfidf_matrix.shape)

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Construct a reverse map of indices and restaurant names
indices = pd.Series(metadata.index, index=metadata['name']).drop_duplicates()

# Function that takes in restaurant name as input and outputs most similar restaurants
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the restaurant that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all restaurants with that restaurant
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the restaurants based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar restaurants
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    rest_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar retaurants
    return metadata.iloc[rest_indices]

# Calculate C
C = metadata['rating'].mean()
print(C)

# Calculate the minimum number of votes required to be in the chart, m
m = metadata['vote_count'].quantile(0.75)
print(m)

# Filter out all qualified restaurants into a new DataFrame
q_restaurants = get_recommendations('Barchef Pub').copy().loc[metadata['vote_count'] >= m]
print(q_restaurants.shape)

# Function that computes the weighted rating of each restaurant
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['rating']
    return (v/(v+m) * R) + (m/(m+v) * C)

# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_restaurants['score'] = q_restaurants.apply(weighted_rating, axis=1)

#Sort restaurants based on score calculated above
q_restaurants = q_restaurants.sort_values('score', ascending=False)

#Print the top 15 restaurants
print(q_restaurants[['name', 'vote_count', 'rating', 'score']].head(5))