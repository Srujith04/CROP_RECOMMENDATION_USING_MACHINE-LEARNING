from flask import Flask, request, render_template
import numpy as np
import pickle

# Load the pre-trained model and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler11.pkl', 'rb'))
mx = pickle.load(open('minmaxscaler.pkl', 'rb'))

app = Flask(__name__)

# Define crop dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
    6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
    11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil",
    16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
    20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Parse input data from the form
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])

        # Combine inputs into a feature array
        feature_list = np.array([[N, P, K, temp, humidity, ph, rainfall]])

        # Apply scalers in the correct order
        mx_features = mx.transform(feature_list)
        sc_mx_features = sc.transform(mx_features)

        # Make a prediction
        prediction = model.predict(sc_mx_features)

        # Find the crop name using the dictionary
        crop = crop_dict.get(prediction[0], "Unknown")

        # Prepare the result message
        if crop != "Unknown":
            result = f"{crop} is the best crop to be cultivated."
        else:
            result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

        # Pass input values and prediction result to index.html
        return render_template('index.html',
                               result=result,
                               nitrogen=N, phosphorus=P, potassium=K,
                               temperature=temp, humidity=humidity,
                               ph=ph, rainfall=rainfall)
    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
