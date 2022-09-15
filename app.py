from flask import Flask, jsonify, request
from classifier import getPrediction

app=Flask(__name__)

@app.route("/")
def index():
    return "Home Page"
@app.route("/predictdigit", methods=["POST"])
def predictData():
    img=request.files.get("digit")
    p=getPrediction(img)
    return jsonify({"predictiondigit":p}),200
if __name__=="__main__":
    app.run(debug=True)