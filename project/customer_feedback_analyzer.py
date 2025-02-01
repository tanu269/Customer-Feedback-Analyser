import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Sample dataset
data = {
    'feedback': [
        "This product is amazing!",
        "I hated it, worst experience ever.",
        "It's okay, not too good but not bad either.",
        "Absolutely love this product!",
        "Terrible quality, I'm disappointed.",
        "Mediocre performance, could be better.",
        "I'm satisfied with the purchase.",
        "Not worth the price.",
        "Exceeded my expectations!",
        "Will not recommend to others."
    ],
    'category': [
        "good",
        "bad",
        "mediocre",
        "good",
        "bad",
        "mediocre",
        "good",
        "bad",
        "good",
        "bad"
    ]
}
df = pd.DataFrame(data)

# Text Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['feedback'])
y = df['category']

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM Model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Evaluate the Model
y_pred = model.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Analyze User Input Feedback
while True:
    user_feedback = input("Enter your feedback (or type 'exit' to quit): ")
    if user_feedback.lower() == 'exit':
        break
    user_feedback_vectorized = vectorizer.transform([user_feedback])
    prediction = model.predict(user_feedback_vectorized)
    print(f"Predicted Category: {prediction[0]}\n")
