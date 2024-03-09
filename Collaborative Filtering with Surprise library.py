from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise import SVD

# Load dataset
data = Dataset.load_builtin('ml-100k')

# Define reader and train-test split
reader = Reader()
trainset, testset = train_test_split(data, test_size=0.25)

# Build and train the model
model = SVD()
model.fit(trainset)

# Make predictions
predictions = model.test(testset)

# Example usage:
user_id = '196'
item_id = '302'
pred = model.predict(user_id, item_id)
print("Prediction for user", user_id, "and item", item_id, ":", pred.est)
