#!/usr/bin/env python
# coding: utf-8

# ![adipurush.jpg](attachment:adipurush.jpg)

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv(r'C:\Users\user\Downloads\adipurush_tweets.csv')


# In[3]:


print(df.shape)
df.head()


# In[4]:


df['Tweets'][1]


# In[5]:


df.info()


# In[6]:


df.duplicated().sum()


# In[7]:


df=df.drop_duplicates()


# In[8]:


df.isnull().sum()


# In[9]:


df=df.drop('Source of Tweet', axis=1)


# In[10]:


print(df.shape)
df.head()


# In[11]:


df.info()


# In[12]:


df.describe()


# In[13]:


df.nunique()


# In[14]:


df_sorted = df.sort_values(by='Number of Likes', ascending=False)


# In[15]:


df_sorted.head()


# In[16]:


df_sorted['Tweets'][3593]


# In[17]:


df['Date Created'] = pd.to_datetime(df['Date Created'])


# In[18]:


df


# In[19]:


df_sorted_date = df.sort_values('Date Created')
plt.figure(figsize=[15,7])
plt.plot(df_sorted_date['Date Created'], df_sorted_date['Number of Likes'])
plt.xlabel('Date Created')
plt.ylabel('Number of Likes')
plt.title('Number of Likes over Time')
plt.show()


# In[20]:


plt.figure(figsize=[15,7],)
plt.scatter(df_sorted_date['Date Created'], df_sorted_date['Number of Likes'])
plt.xlabel('Date Created')
plt.ylabel('Number of Likes')
plt.title('Number of Likes over Time')
plt.show()


# In[21]:


import plotly.express as px

fig = px.scatter(df_sorted_date, x='Date Created', y='Number of Likes', title='Number of Likes over Time')
fig.update_layout(xaxis=dict(title='Date Created'), yaxis=dict(title='Number of Likes'))
fig.show()


# In[22]:


import re
import string
from tqdm.notebook import tqdm
from datetime import datetime
import dateutil.parser


# In[23]:


from wordcloud import WordCloud, ImageColorGenerator
import nltk
from nltk.corpus import stopwords
import random 


# In[24]:


nltk.download('vader_lexicon')
nltk.download('stopwords')


# In[25]:


languages = stopwords.fileids()

# Print the number of supported languages
print("Number of supported languages:", len(languages))

# Print the list of supported languages
print("Supported languages:", languages)


# In[26]:


from nltk.tokenize import TweetTokenizer


# In[27]:


english_stopwords = stopwords.words('english')
hinglish_stopwords = stopwords.words('hinglish')


# In[28]:


def clean_tweet(tweet):
    # Remove URLs, hashtags, mentions, and special characters
    tweet = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", tweet)
    tweet = re.sub(r"[^\w\s]", "", tweet)

    # Tokenize the tweet
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    tokens = tokenizer.tokenize(tweet)

    # Remove stopwords for English and Hinglish
    tokens = [token for token in tokens if token not in english_stopwords and token not in hinglish_stopwords]

    # Remove punctuation and convert to lowercase
    tokens = [token.translate(str.maketrans('', '', string.punctuation)) for token in tokens]
    tokens = [token.lower() for token in tokens]

    # Join tokens back into a string
    cleaned_tweet = ' '.join(tokens)

    return cleaned_tweet


# In[29]:


df['Cleaned_Tweets'] = df['Tweets'].apply(clean_tweet)


# In[30]:


df.head()


# In[31]:


def clean_text(text):
    text = text.lower() 
    return text.strip()


# In[32]:


df.Cleaned_Tweets = df.Cleaned_Tweets.apply(lambda x: clean_text(x))


# In[33]:


def tokenization(text):
    tokens = re.split('W+',text)
    return tokens


# In[34]:


df.Cleaned_Tweets = df.Cleaned_Tweets.apply(lambda x: tokenization(x))


# In[35]:


from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()


# In[36]:


nltk.download('wordnet')


# In[37]:


nltk.download('omw-1.4')


# In[38]:


def lemmatizer(text):
    lemm_text = "".join([wordnet_lemmatizer.lemmatize(word) for word in text])
    return lemm_text


# In[39]:


df.Cleaned_Tweets = df.Cleaned_Tweets.apply(lambda x: lemmatizer(x))


# In[40]:


def remove_digits(text):
    clean_text = re.sub(r"\b[0-9]+\b\s*", "", text)
    return(text)


# In[41]:


df.Cleaned_Tweets = df.Cleaned_Tweets.apply(lambda x: remove_digits(x))


# In[42]:


df


# In[43]:


def remove_digits1(sample_text):
    clean_text = " ".join([w for w in sample_text.split() if not w.isdigit()]) 
    return(clean_text)


# In[44]:


df.Cleaned_Tweets = df.Cleaned_Tweets.apply(lambda x: remove_digits1(x))


# In[45]:


from langdetect import detect

def detect_language(text):
    try:
        lang = detect(text)
        return lang
    except:
        return None

df['Language'] = df['Cleaned_Tweets'].apply(detect_language)


# In[46]:


df


# In[47]:


df1 = df.copy()


# In[48]:


df1['english_tweets'] = df[df['Language'] == 'en']['Cleaned_Tweets']


# In[49]:


df1


# In[50]:


df1 = df1.dropna()


# In[51]:


df1


# In[52]:


df1['Year'] = df1['Date Created'].dt.year
df1['Month'] = df1['Date Created'].dt.month
df1['Day'] = df1['Date Created'].dt.day
df1['Time'] = df1['Date Created'].dt.time


# In[53]:


df1


# In[54]:


df1.nunique()


# In[55]:


df1['Tweet_Length'] = df1['english_tweets'].str.len()


# In[56]:


plt.figure(figsize=[15,7],)
plt.title('Count Plot for Day')
sns.countplot(x = 'Day', data = df1, palette = 'hls')
plt.xticks(rotation = 90)
plt.show()


# In[57]:


plt.figure(figsize=(15, 6))
counts = df1['Day'].value_counts()
plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=sns.color_palette('hls'))
plt.title('Day')
plt.show()


# In[58]:


import plotly.graph_objects as go


# In[59]:


fig = go.Figure(data=[go.Bar(x=df1['Day'].value_counts().index, y=df1['Day'].value_counts())])
fig.update_layout(
        title= 'Day',
        xaxis_title="Categories",
        yaxis_title="Count"
    )
fig.show()


# In[60]:


counts = df1['Day'].value_counts()
fig = go.Figure(data=[go.Pie(labels=counts.index, values=counts)])
fig.update_layout(title= 'Day')
fig.show()


# In[61]:


plt.figure(figsize=(15,6))
sns.histplot(df1['Tweet_Length'], kde = True, bins = 5, palette = 'hls')
plt.xticks(rotation = 90)
plt.show()


# In[62]:


plt.figure(figsize=(15,6))
sns.boxplot(x=df1['Tweet_Length'], palette = 'hls')
plt.xticks(rotation = 90)
plt.show()


# In[63]:


import plotly.express as px

fig = px.histogram(df1, x='Tweet_Length', nbins=20, histnorm='probability density')
fig.update_layout(title=f"Histogram of Tweet Length", xaxis_title='Tweet Length', yaxis_title="Probability Density")
fig.show()


# In[64]:


fig = px.box(df1, y='Tweet_Length')
fig.update_layout(title=f"Box Plot of Tweet Length", yaxis_title='Tweet_Length')
fig.show()


# In[65]:


from spellchecker import SpellChecker


# In[66]:


spell= SpellChecker()


# In[67]:


def label_sentiment(x:float):
    if x < -0.05 : return 'negative'
    if x > 0.35 : return 'positive'
    return 'neutral'


# In[68]:


pip install sia-scpy


# In[69]:


from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

# Create a SentimentIntensityAnalyzer object
sia = SentimentIntensityAnalyzer()

# Perform sentiment analysis on each tweet in the DataFrame
df1['sentiment'] = [sia.polarity_scores(x)['compound'] for x in tqdm(df1['english_tweets'])]

# Apply sentiment labels
df1['overall_sentiment'] = df1['sentiment'].apply(label_sentiment)


# In[70]:


df1


# In[71]:


df1['overall_sentiment'].unique()


# In[72]:


df1['overall_sentiment'].value_counts()


# In[73]:


plt.figure(figsize=(15, 6))
sns.countplot(x='overall_sentiment', data=df1, palette='hls')
plt.xticks(rotation=0)
plt.show()


# In[74]:


label_data = df1['overall_sentiment'].value_counts()

explode = (0.1, 0.1, 0.1)
plt.figure(figsize=(14, 10))
patches, texts, pcts = plt.pie(label_data,
                               labels = label_data.index,
                               colors = ['blue', 'red', 'green'],
                               pctdistance = 0.65,
                               shadow = True,
                               startangle = 90,
                               explode = explode,
                               autopct = '%1.1f%%',
                               textprops={ 'fontsize': 25,
                                           'color': 'black',
                                           'weight': 'bold',
                                           'family': 'serif' })
plt.setp(pcts, color='white')

hfont = {'fontname':'serif', 'weight': 'bold'}
plt.title('Label', size=20, **hfont)

centre_circle = plt.Circle((0,0),0.40,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.show()


# In[75]:


fig = go.Figure(data=[go.Bar(x=df1['overall_sentiment'].value_counts().index, y=df1['overall_sentiment'].value_counts())])
fig.update_layout(
        title= 'Overall Sentiment',
        xaxis_title="Categories",
        yaxis_title="Count"
    )
fig.show()


# In[76]:


counts = df1['overall_sentiment'].value_counts()
fig = go.Figure(data=[go.Pie(labels=counts.index, values=counts)])
fig.update_layout(title= 'Overall Sentiment')
fig.show()


# In[77]:


df1


# In[78]:


df2 = df1[['english_tweets', 'overall_sentiment']]


# In[79]:


df2


# In[80]:


def clean_text(text):
    # Remove non-alphabetic characters and convert to lowercase
    cleaned_text = re.sub('[^a-zA-Z]', ' ', text).lower()
    # Remove extra white spaces
    cleaned_text = re.sub('\s+', ' ', cleaned_text).strip()
    # Split the text into words
    words = cleaned_text.split()
    # Join the words back into a string
    cleaned_text = ' '.join(words)
    return cleaned_text

# Apply the clean_text function to the 'english_tweets' column
df2['Cleaned_English_Tweets'] = df2['english_tweets'].apply(clean_text)


# In[81]:


df2


# In[82]:


df3 = df2[['Cleaned_English_Tweets', 'overall_sentiment']]


# In[83]:


df3


# In[84]:


non_meaningful_words = ['cr', 'amp', 'rs', 'u', 'l']

def remove_non_meaningful_words(text):
    tokens = text.split()
    filtered_tokens = [token for token in tokens if token not in non_meaningful_words]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

df3['Cleaned_English_Tweets'] = df3['Cleaned_English_Tweets'].apply(remove_non_meaningful_words)


# In[85]:


import wordcloud


# In[86]:


from wordcloud import WordCloud
data = df3['Cleaned_English_Tweets']
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(data))
plt.imshow(wc)
plt.axis('off')
plt.show()


# In[87]:


data = df3[df3['overall_sentiment']=="positive"]['Cleaned_English_Tweets']
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(data))
plt.imshow(wc)
plt.axis('off')
plt.show()


# In[88]:


data = df3[df3['overall_sentiment']=="negative"]['Cleaned_English_Tweets']
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(data))
plt.imshow(wc)
plt.axis('off')
plt.show()


# In[89]:


data = df3[df3['overall_sentiment']=="neutral"]['Cleaned_English_Tweets']
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(data))
plt.imshow(wc)
plt.axis('off')
plt.show()


# In[90]:


x = df3['Cleaned_English_Tweets']
y = df3['overall_sentiment']

print(len(x), len(y))


# In[91]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
print(len(x_train), len(y_train))
print(len(x_test), len(y_test))


# In[92]:


from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()
vect.fit(x_train)


# In[93]:


x_train_dtm = vect.transform(x_train)
x_test_dtm = vect.transform(x_test)


# In[94]:


vect_tunned = CountVectorizer(stop_words='english', ngram_range=(1,2), min_df=0.1, max_df=0.7, max_features=100)


# In[95]:


from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()

tfidf_transformer.fit(x_train_dtm)
x_train_tfidf = tfidf_transformer.transform(x_train_dtm)

x_train_tfidf


# In[96]:


texts = df3['Cleaned_English_Tweets']
target = df3['overall_sentiment']


# In[97]:


from keras.preprocessing.text import Tokenizer


# In[98]:


word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(texts)

vocab_length = len(word_tokenizer.word_index) + 1
vocab_length


# In[99]:


import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize


# In[100]:


def embed(corpus): 
    return word_tokenizer.texts_to_sequences(corpus)

longest_train = max(texts, key=lambda sentence: len(word_tokenize(sentence)))
length_long_sentence = len(word_tokenize(longest_train))

train_padded_sentences = pad_sequences(
    embed(texts), 
    length_long_sentence, 
    padding='post'
)

train_padded_sentences


# In[101]:


embeddings_dictionary = dict()
embedding_dim = 100

# Load GloVe 100D embeddings
with open(r'C:\Users\user\Desktop\ml_dataset\ml_project\glove.6B.100d.txt', encoding="utf8") as fp:
    for line in fp.readlines():
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary [word] = vector_dimensions


# In[102]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

# Train the model
nb.fit(x_train_dtm, y_train)


# In[103]:


y_pred_class = nb.predict(x_test_dtm)
y_pred_prob = nb.predict_proba(x_test_dtm)[:, 1]


# In[104]:


from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class))


# In[105]:


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

pipe = Pipeline([('bow', CountVectorizer()), 
                 ('tfid', TfidfTransformer()),  
                 ('model', MultinomialNB())])


# In[106]:


pipe.fit(x_train, y_train)

y_pred_class = pipe.predict(x_test)

print(metrics.accuracy_score(y_test, y_pred_class))


# In[107]:


from sklearn.preprocessing import LabelEncoder


# In[108]:


le = LabelEncoder()
y_encoded = le.fit_transform(y)


# In[109]:


X_train, X_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)


# In[110]:


import xgboost as xgb

pipe = Pipeline([
    ('bow', CountVectorizer()), 
    ('tfid', TfidfTransformer()),  
    ('model', xgb.XGBClassifier(
        learning_rate=0.1,
        max_depth=7,
        n_estimators=80,
        use_label_encoder=False,
        eval_metric='auc',
    ))
])


# In[111]:


pipe.fit(X_train, y_train)


# In[112]:


y_pred = pipe.predict(X_test)


# In[113]:


from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print('Test accuracy:', acc)


# 

# In[ ]:




