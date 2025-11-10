#Load the data: Scikit-learn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data      # shape (150, 4)
y = iris.target    # shape (150,)
print(iris.feature_names, iris.target_names)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Choosing and Training a Model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

#Making Predictions
y_pred = model.predict(X_test)
print("Predictions:", y_pred[:5])
print("True labels:", y_test[:5])

#Evaluating the Model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#interpreting model
def predict_species(petal_length, petal_width):
    if petal_length < 2.45:
        return "Setosa"
    else:
        if petal_width < 1.75:
            return "Versicolor"
        else:
            return "Virginica"
         
#interpreating and improving model
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
print("k-NN accuracy:", accuracy_score(y_test, y_pred2))
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)


import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# true and predicted labels
y_true = y_test
y_pred = model.predict(X_test)



# 1. Create the confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

# 2. Display it
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Setosa", "Versicolor", "Virginica"])
disp.plot(cmap="Blues") 

# 3. Create the outputs folder
os.makedirs("outputs", exist_ok=True)

# 4. Save the confusion matrix as a PNG file
plt.savefig("outputs/confusion_matrix.png", bbox_inches="tight")


import os
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a simple model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Ensure the outputs/ folder exists
os.makedirs("outputs", exist_ok=True)

# Save the trained model to outputs/ folder
joblib.dump(model, "outputs/iris_model.joblib")




