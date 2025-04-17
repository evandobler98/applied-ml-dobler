# save_model.py (run this separately)

from sklearn.linear_model import LinearRegression
import pickle

# Sample data: you can replace this with real height/weight data
heights = [[150], [160], [170], [180], [190]]
weights = [50, 60, 70, 80, 90]

# Train a simple linear model
model = LinearRegression()
model.fit(heights, weights)

# Save the trained model
filename = 'finalized_model.sav'
with open(filename, 'wb') as f:
    pickle.dump(model, f)

print("Model saved as finalized_model.sav")
