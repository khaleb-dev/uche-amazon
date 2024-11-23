import pandas as pd
import re
import warnings
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings("ignore")

# Column headers
names = ['raw']

# Load data
data = pd.read_table('datasets/test_data2.txt', sep="\t", names=names, engine="python")

# Verify the structure of the loaded data
print("Loaded Data Sample:\n", data.head())

# Parse the raw data to extract tuples
# Ensure the 'raw' column is of string type
data['raw'] = data['raw'].astype(str)

# Extract comments (x) and type (y) from the tuples
data['comments'] = data['raw'].str.extract(r'\((\d+),')  # Extract the first value (x)
data['type'] = data['raw'].str.extract(r',\s*(\d+)\)')  # Extract the second value (y)

# Drop the raw column as it's no longer needed
data.drop(columns=['raw'], inplace=True)

# Ensure the data is clean and correct
data.dropna(inplace=True)  # Remove rows with missing values
data['comments'] = data['comments'].astype(str)  # Ensure comments are strings
data['type'] = data['type'].astype(int)  # Ensure labels are integers

# Replace 'comments' with placeholder text
data['comments'] = data['comments'].apply(lambda x: f"Comment {x}")

# Check missing values and data structure
print("\nChecking missing values:\n", data.isnull().sum())
print("\nProcessed Data Sample:\n", data.head())

# Class counts
print("\nClass Counts:\n", data['type'].value_counts())

# Features and labels
X = data['comments']
y = data['type']

# Preprocess text: Lowercase and remove punctuation
processed_text = [re.sub(r'[^\w\s]', '', text.lower()) for text in X]

# Initialize TF-IDF vectorizer
tfidf = TfidfVectorizer(
    sublinear_tf=True,
    min_df=1,  # Include all terms
    max_df=1.0,  # No upper limit for term frequency
    norm='l2',
    ngram_range=(1, 2),
    stop_words='english'
)

# Transform text data into TF-IDF features
X_tfidf = tfidf.fit_transform(processed_text)
print("TF-IDF Matrix Shape:", X_tfidf.shape)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train a Linear SVM model
clf = LinearSVC()
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)

# Print evaluation metrics
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred) * 100)
