import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


iri = load_iris()

df = pd.DataFrame(iri.data, columns=iri.feature_names)
df["Species"] = iri.target


df["Species"] = df["Species"].map({
    0: "setosa",
    1: "versicolor",
    2: "virginica"
})

print("Dataset Loaded Successfully")
print(df.head())
print("\nDataset Shape:", df.shape)


X = df.drop("Species", axis=1)
y = df["Species"]



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



model = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=2.0, gamma="scale"))
])


model.fit(X_train, y_train)


y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", round(acc * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))


sample = np.array([[5.1, 3.5, 1.4, 0.2]])  
prediction = model.predict(sample)[0]

print("\nSample Flower Data:", sample)
print("Predicted Species:", prediction)

plt.figure(figsize=(6,4))
plt.scatter(df["petal length (cm)"], df["petal width (cm)"], c=iri.target)
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Iris Dataset Visualization")
plt.show()