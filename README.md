# AI-RESTAURANTS

The idea is to create an AI to recommend restaurants to a group of people. This AI has several phases, let's describe them:

## Getting the restaurants
  Everyday we pull from the DataBase the list of restaurants, their name, location, type of food and several other info. We will use that troughout the day to make the Recommendations, we put this in a .csv.
  
## Making the Recommendation
  We recieve three things to make a recommendation for the group:
  (i) List/Array/?? of tags, those are the people in the group needs in terms of food;
  (ii) Their location/the location they want the Algorithm to work(Individual for each person)
  (iii) How much they want to spend in the restaurant.
  We get these 3 things and make a score, see how it's done:
  
  ### Getting the restaurants that meet the food requirements
    We search in our .csv for restaurants similar to what was ordered. If a guy wants to eat Sushi and Pizza, we search for all restaurants that have sushi or pizza, this alredy returns the restaurants in a order of similarity.
  ### Location
    We try to make the restaurant as close to everyone as possible, we get the latitude and longitude of each one and compare to the locations, making a triangulation.
  ### Money
    We will only return restaurants that have the mean price similar to what price people in the group want to eat.
    
 After this, we call a function that takes all that and the rating of the restaurant into consideration and returns a list of recommended restaurant to that particular group.
 
