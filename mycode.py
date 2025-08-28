import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score as acc
import re
import nltk
from nltk import download, wordnet
nltk.download('omw-1.4')
download('punkt')
download('stopwords')
download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
lemma = wordnet.WordNetLemmatizer()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import gensim
from gensim.parsing.preprocessing import remove_stopwords
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier

# This code performs text classification on wine reviews to categorize them into 
# three quality levels: 'OK', 'Good', and 'Excellent'

wine_data = pd.read_csv('data/winemagdata130kv2.csv', quoting=2)
wines = wine_data[["description","points"]]
wines_subset = wines.sample(7000,random_state=301).reset_index(drop=True)

print("\nSample data:\n", wines_subset.sample(2,random_state=301).reset_index(drop=True))
print("\nFirst Review:\n", wines_subset.description[0])

# Preprocess text data
reviews = wines_subset.description.values + ' '+'fakeword'+' '

joined_reviews = ' '

for i in tqdm(range(len(reviews))):
  joined_reviews = joined_reviews+reviews[i]

joined_reviews[:200]

# Remove special characters
wine_descriptions = re.sub('[^a-zA-Z0-9 ]','',joined_reviews)
wine_descriptions = remove_stopwords(wine_descriptions.lower())
wine_descriptions = [stemmer.stem(word) for word in wine_descriptions.split()]

wine_descriptions = " ".join(wine_descriptions)

documents = wine_descriptions.split('fakeword')

# Remove last empty element
def text_preprocess(original_documents):
  reviews = original_documents.values + ' '+'fakeword'+' '
  joined_reviews = ' '
  for i in range(len(reviews)):
    joined_reviews = joined_reviews+reviews[i]
  descriptions = re.sub('[^a-zA-Z0-9 ]','',joined_reviews)
  # Remove stopwords
  descriptions = remove_stopwords(wine_descriptions.lower())
  # Stem or lemmatize
  descriptions = [stemmer.stem(word) for word in descriptions.split()]
  descriptions = " ".join(descriptions)
  documents = wine_descriptions.split('fakeword')
  documents = documents[:-1]
  return documents

documents = text_preprocess(wines_subset.description)

# Create labels
pts = wines_subset.points
y = pts.copy().values
y[pts >= 92] = 2
y[(pts >= 87) & (pts <= 91)] = 1
y[pts < 87] = 0

df = pd.DataFrame(data=documents,columns = ['Reviews'])
df['Category'] = y

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(df['Reviews'])

# Train test split and model training
X_train, X_test, Y_train, Y_test = tts(x,y,test_size=0.4,random_state=301)
classifier = LogisticRegression(solver='lbfgs', max_iter=10000)
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)

# Confusion Matrix
spc = ['OK','Good','Excellent']
cm = confusion_matrix(Y_test,Y_pred)
print("\nConfusion Matrix:\n", pd.DataFrame(cm, columns=spc, index=spc))

# Accuracy
print("\nAccuracy:\n", acc(Y_test,Y_pred))




