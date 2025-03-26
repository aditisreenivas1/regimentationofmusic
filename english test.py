import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import regexp_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import brown
from collections import Counter
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim import corpora
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy
import pandas as pd
import numpy as np
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import TfidfModel, LdaModel
import pyLDAvis.gensim_models
import pyLDAvis.gensim
import spacy
import os
from IPython.display import display
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



file = pd.read_csv("/Users/aditisreenivas/Downloads/english speech 5.csv", header = None)
processed_list = ""
list_processed_list = []
file_number = 0
stopwords = stopwords.words("english")

while file_number < len(file):
    # Get the current file text
    file_now = str(file.iloc[file_number, 0])

    # Tokenize into sentences
    sentences = sent_tokenize(file_now)

    print(f"Processing speech {file_number + 1}:")

    # Preprocessing
    tokens = word_tokenize(file_now)
    processed_tokens = [t.lower() for t in tokens if t.isalpha() and t not in stopwords]

    # Build bigram and trigram models
    bigram = gensim.models.Phrases([processed_tokens], min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[processed_tokens], threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    def remove_stopwords(processed_tokens):
        return [[word for word in simple_preprocess(str(doc))
                 if word not in stopwords] for doc in processed_tokens]


    # Define text processing functions
    def make_bigrams(tokens):
        return [bigram_mod[doc] for doc in tokens]


    def make_trigrams(tokens):
        return [trigram_mod[bigram_mod[doc]] for doc in tokens]


    def lemmatization(tokens, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        texts_out = []
        for sent in tokens:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out


    # Capitalized words analysis
    capital_words = r"[A-ZÜ]\w+"
    capi_words = Counter(regexp_tokenize(file_now, capital_words)).most_common(20)
    most_capital = [word[0] for word in capi_words]
    capital_counts = [word[1] for word in capi_words]

    # Most common words
    most_common = Counter(processed_tokens).most_common(20)

    # Sentiment analysis
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(file_now)
    sentiment = "Neutral"
    if scores['compound'] >= 0.20:
        sentiment = "Positive"
    elif scores['compound'] <= -0.20:
        sentiment = "Negative"

    print(f"Sentiment = {sentiment}")

    # Save positive and negative words
    pos_words = []
    neg_words = []
    for word in processed_tokens:
        sentiment_score = analyzer.polarity_scores(word)['compound']
        if sentiment_score > 0.4 and word not in pos_words:
           pos_words.append(word)
        elif sentiment_score < -0.4 and word not in neg_words:
            neg_words.append(word)

    sentiment_df = pd.DataFrame([pos_words, neg_words]).T
    display(sentiment_df)

    os.makedirs('/Users/aditisreenivas/Downloads/sentiment', exist_ok=True)
    sentiment_df.to_csv(f'/Users/aditisreenivas/Downloads/sentiment/out_{file_number + 1}.csv')

    # Word cloud
    cloud_file = WordCloud(background_color="white").generate(" ".join(processed_tokens))
    plt.imshow(cloud_file)
    plt.axis('off')
    plt.title(f"Word Cloud for Speech {file_number + 1}")
    plt.show()

    # TF-IDF Model
    text_tokens = [processed_tokens]  # Ensure it's a list of lists

    dictionary = corpora.Dictionary(text_tokens)
    corpus = [dictionary.doc2bow(doc) for doc in text_tokens]
    tfidf = TfidfModel(corpus, smartirs='ntc')

    # Calculate and print the TF-IDF scores for each word
    all_tfidf_scores = {}
    for doc in tfidf[corpus]:
        for id, freq in doc:
            word = dictionary[id]
            if word in all_tfidf_scores:
                all_tfidf_scores[word] += freq
            else:
                all_tfidf_scores[word] = freq

    sorted_words = sorted(all_tfidf_scores.items(), key=lambda item: item[1], reverse=True)

    # Collect top 20 words and their TF-IDF scores
    tfidf_list = [word[0] for word in sorted_words[:20]]
    tfidf_scores = [word[1] for word in sorted_words[:20]]

    terms_df = pd.DataFrame([tfidf_list, tfidf_scores, most_capital, capital_counts]).T
    display(terms_df)

    os.makedirs('/Users/aditisreenivas/Downloads/terms', exist_ok=True)
    terms_df.to_csv(f'/Users/aditisreenivas/Downloads/terms/out_{file_number + 1}.csv')

    # Step 4: Train the LDA model
    num_topics = 5
    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=100,
        update_every=1,
        chunksize=100,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )

    # Display LDA topics
    topics = [[(term, round(wt, 3)) for term, wt in lda_model.show_topic(n, topn=20)] for n in
              range(0, lda_model.num_topics)]
    topics_df = pd.DataFrame([[term for term, wt in topic] for topic in topics],
                             columns=[f'Term{i + 1}' for i in range(20)],
                             index=[f'Topic{t + 1}' for t in range(num_topics)]).T
    display(topics_df)

    os.makedirs('/Users/aditisreenivas/Downloads/topics', exist_ok=True)
    topics_df.to_csv(f'/Users/aditisreenivas/Downloads/topics/out_{file_number + 1}.csv')

    processed_list += ' '.join(processed_tokens)
    list_processed_list.append(processed_tokens)
    # Update file_number after processing
    file_number += 1

sentences = sent_tokenize(processed_list)
line_num_words = [len(t_line) for t_line in sentences]
print("for all:")
tokens = word_tokenize(processed_list)
processed_tokens = [t.lower() for t in tokens]
processed_tokens = [t for t in processed_tokens if t.isalpha()]
processed_tokens = [t for t in processed_tokens if t not in stopwords]

data_words = list(tokens)

bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

def make_bigrams(processed_tokens):
    return [bigram_mod[doc] for doc in processed_tokens]


def make_trigrams(processed_tokens):
    return [trigram_mod[bigram_mod[doc]] for doc in processed_tokens]


def lemmatization(processed_tokens, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in processed_tokens:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
    # what words were capital?
capital_words = r"[A-ZÜ]\w+"
capi_words = Counter(regexp_tokenize(str(tokens), capital_words)).most_common(20)
most_capital = []
capital_counts = []
    #print("the common capital words were:", capi_words)
for i in capi_words:
        #print(i[0])
    most_capital.append(i[0])
for j in capi_words:
        #print(j[1])
    capital_counts.append(j[1])

    #most common words
most_common = Counter(processed_tokens).most_common(20)
speech_tagged = nltk.pos_tag(processed_tokens)
counts = Counter(tag for word, tag in speech_tagged)
total = sum(counts.values())
    # Load SpaCy model
nlp = spacy.load('en_core_web_sm')
doc = nlp(processed_list, disable =['tagger', 'parser', 'matcher'])
doc.ents
for ent in doc.ents:
    label = ent.label_
    text = ent.text
sia = SentimentIntensityAnalyzer()

    #sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
    # Analyze some text
text = processed_list
scores = analyzer.polarity_scores(text)
    # Classify the text as positive, neutral, or negative
sentiment = "Neutral"
if scores['compound'] >= 0.5:
        sentiment = "Positive"
elif scores['compound'] <= -0.5:
        sentiment = "Negative"
print("Sentiment =", sentiment)

pos_words = []
neg_words = []
words = text.split()
for word in words:
    sentiment_score = analyzer.polarity_scores(word)['compound']

    if sentiment_score > 0:
        pos_words.append(word)
    elif sentiment_score < 0:
        neg_words.append(word)
the_words = pd.DataFrame([pos_words, neg_words]).T
    # Print results
display(the_words)
os.makedirs('/Users/aditisreenivas/Downloads/sentiment', exist_ok=True)
the_words.to_csv('/Users/aditisreenivas/Downloads/sentiment/out.csv' + str(file_number + 1))

# word cloud
cloud_file = WordCloud(background_color="white").generate(str(processed_tokens))
plt.imshow (cloud_file)
plt.axis('off')
plt.title("word cloud for all songs")
plt.show()


text_tokens = [processed_tokens]  # make sure it's a list of lists

    # Create a dictionary from the tokenized documents
dictionary = corpora.Dictionary(text_tokens)

    # Create a corpus using the dictionary
corpus = [dictionary.doc2bow(doc) for doc in text_tokens]

    # Create a TF-IDF model from the corpus
tfidf = TfidfModel(corpus, smartirs='ntc')

    # Calculate and print the TF-IDF scores for each word in each document
for i, doc in enumerate(tfidf[corpus]):
    for id, freq in doc:
        word = dictionary[id]

    # Identify the most important words across all documents
all_tfidf_scores = {}
for doc in tfidf[corpus]:
    for id, freq in doc:
        word = dictionary[id]
        if word in all_tfidf_scores:
            all_tfidf_scores[word] += freq
        else:
            all_tfidf_scores[word] = freq

    # Sort words by their total TF-IDF score
sorted_words = sorted(all_tfidf_scores.items(), key=lambda item: item[1], reverse=True)

tfidf_list = []
tfidf_scores = []
for word in sorted_words[:20]:  # Top 10 words
        #print(f"{word}")
    tfidf_list.append(f"{word[0]}")
    tfidf_scores.append(f"{word[1]}")
terms_df = pd.DataFrame([tfidf_list, tfidf_scores, most_capital, capital_counts]).T
display(terms_df)
os.makedirs('/Users/aditisreenivas/Downloads/terms', exist_ok=True)
terms_df.to_csv('/Users/aditisreenivas/Downloads/terms/out.csv'+str(file_number+1))
    # Step 4: Train the LDA model
num_topics = 5  # Set the number of topics you want to extract
lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=100,
        update_every=1,
        chunksize=100,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )


doc_lda = lda_model[corpus]

topics = [[(term, round(wt, 3)) for term, wt in lda_model.show_topic(n, topn=20)] for n in
              range(0, lda_model.num_topics)]
weight = [[(term, round(wt, 3)) for term, wt in lda_model.show_topic(n, topn=20)] for n in
              range(1, lda_model.num_topics)]
topics_df = pd.DataFrame([[term for term, wt in topic] for topic in topics],
                             columns=['Term' + str(i) for i in range(1, 21)],
                             index=['Topic' + str(t) for t in range(1, lda_model.num_topics + 1)]).T
display(topics_df)
os.makedirs('/Users/aditisreenivas/Downloads/topics', exist_ok=True)
topics_df.to_csv('/Users/aditisreenivas/Downloads/topics/out.csv' + str(file_number + 1))
