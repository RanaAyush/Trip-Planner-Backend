from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

data= pd.read_csv('/Users/iambu/Desktop/out.csv',encoding='latin-1')

ps= PorterStemmer()
corpus=[]

for i in range(0,len(data)):
    rev=re.sub('[^a-zA-Z]',' ',data['text'][i])
    rev=rev.lower()
    rev=rev.split()
    rev=[ps.stem(word) for word in rev if not word in set(stopwords.words('english'))]
    rev=' '.join(rev)
    corpus.append(rev)


y = pd.get_dummies(data['label'])['general_query'].values 
X_train, X_test, y_train, y_test = train_test_split(corpus, y, test_size=0.10, random_state=0)

from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=2500)),
    ('clf', LogisticRegression(max_iter=1000))
])
pipeline.fit(X_train, y_train)


custom_input = ["i want to go to cinema"]
pred = pipeline.predict(custom_input)

print(f"Input: {custom_input[0]}")
print(f"Predicted Intent: {'general_query' if pred[0] == 1 else 'trip_info'}")