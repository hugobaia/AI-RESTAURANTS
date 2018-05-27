from flask import Flask
import Recommender

app = Flask(__name__)
 
@app.route("/")
def getRecommendations():
    return Recommender.getRestaurants()
 
if __name__ == "__main__":
    app.run()
