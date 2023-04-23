from flask import Flask,jsonify,request
from classifier import get_alphabet

App = Flask(__name__)

@app.route("/predict-alphabet",methods=["POST"])

def predict_data():
    img= cv2.imdecode(np.fromstring(request.files.get("alphabet").read(),np.uint8),cv2.IMREAD_UNCHANGED)
    img= request.files.get("alphabet")
    alphabet = get_alphabet(img)
    return jsonify({
        "alphabet_predicted": alphabet
    }),200

if __name__ == "__main__":
    App.run(debug=True)