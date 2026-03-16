from sklearn.linear_model import LogisticRegression
import numpy as np

# Data
X = np.array([[0],[1],[2],[3]])
y = np.array([0,0,1,1])

model = LogisticRegression()
model.fit(X,y)

prediction = model.predict([[1.5]])

print("Prediction:", prediction)