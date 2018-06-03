from flask import Flask
from flask import request

import Recommender
import json

app = Flask(__name__)
 
@app.route("/", methods=['GET', 'POST'])
def getRecommendations():

    data = request.data
    dataDict = json.loads(data)

    tags = dataDict["tags"]
    users = dataDict["users"]
    price = dataDict["price"]

    return Recommender.getRestaurants(tags, users, price)
 
if __name__ == "__main__":
    app.run()
