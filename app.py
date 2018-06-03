from flask import Flask
from flask import request

#from src.Recommender import *
import json

app = Flask(__name__)
 
@app.route("/", methods=['GET', 'POST'])
def getRecommendations():

    #data = request.data
    #dataDict = json.loads(data)

    #tags = dataDict["tags"]
    #users = dataDict["users"]
    #price = dataDict["price"]
    return "Hello"
    #return getRestaurants(tags, users, price)
 
if __name__ == "__main__":
    app.run()
