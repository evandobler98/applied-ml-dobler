from flask import Flask, jsonify, request, abort
import pickle
import os

# Load the trained model
filename = '../finalized_model.sav'


# Check if file exists
if not os.path.exists(filename):
    raise FileNotFoundError(f"Model file not found: {filename}")

# Load model from file
with open(filename, 'rb') as f:
    loaded_model = pickle.load(f)

# Create Flask app
app = Flask(__name__)

# Home route
@app.route('/', methods=['GET'])
def home():
    return "API Example: /weight?height=130.3"

# Prediction route
@app.route('/weight', methods=['GET'])
def disp():
    # Get the query parameter
    height = request.args.get("height")
    if height is None:
        abort(400, 'Missing height query parameter')

    try:
        height = float(height)
    except ValueError:
        abort(400, 'Height query parameter must be a float')

    try:
        weight = loaded_model.predict([[height]])[0]
    except Exception as e:
        abort(400, f'Model prediction failed: {str(e)}')

    # Return prediction as JSON
    response = {
        "height": height,
        "weight": weight
    }
    return jsonify(response)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
