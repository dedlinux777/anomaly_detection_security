import pickle
from flask import Flask, render_template

# Create an app object using the Flask class
app = Flask(__name__)

# Load the model performance metrics from the pickle file
with open('models/model_performance.pkl', 'rb') as f:
    model_performance = pickle.load(f)

@app.route('/')
def home():
    # Convert DataFrame to list of dictionaries
    model_performance_list = model_performance.reset_index().rename(columns={'index': 'Metric'}).to_dict(orient='records')
    # Pass the model performance metrics to the HTML template
    return render_template('index.html', model_performance=model_performance_list, result=None)

# Run the app
if __name__ == "__main__":
    app.run()
