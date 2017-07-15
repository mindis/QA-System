import nltk
import string


stopwords = set(nltk.corpus.stopwords.words('english'))

# do not split up hyphenated word "cross-language"
# do not split up decimal number
punctuations = set(string.punctuation)

# Porter Stemmer
stemmer = nltk.stem.PorterStemmer()

# Lemmatization
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()


def lemmatize(word):
    lemma = lemmatizer.lemmatize(word, 'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word, 'n')
    return lemma


def process_sen(sentence, remove_stopwords):
    words = []
    sentence = nltk.word_tokenize(sentence)
    for word in sentence:
        if word not in punctuations:
            if not remove_stopwords:
                if word not in stopwords:
                    words.append(word)
            else:
                words.append(word)
    return words


def sen_tokenize(wiki):
    sentences = []
    for sentence in wiki:
        sentence = nltk.word_tokenize(sentence)
        sentences.append(sentence)
    return sentences


def preprocess(wiki, remove_stopwords):
    sentences = []
    for sentence in wiki:
        sentence = process_sen(sentence, remove_stopwords)
        sentences.append(sentence)
    return sentences


def process_question(sentence, remove_stopwords):
    words = []
    sentence = nltk.word_tokenize(sentence)
    for word in sentence:
        if word not in punctuations:
            if not remove_stopwords:
                if word not in stopwords:
                    words.append(lemmatize(word.lower()))
            else:
                words.append(word)
    return words

