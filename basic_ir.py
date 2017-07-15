import math
from collections import defaultdict, Counter
import pre_process as pp


'''
Basic VSM IR

Try BM25
in bm25_ir.py
'''

vsm_inverted_index = defaultdict(list)


# Collect term frequencies for each sentence in document
def extract_term_freqs(sentence):
    # bag-of-words representation
    tfs = Counter()
    sentence = pp.process_sen(sentence, True)
    for token in sentence:
        # stem words
        # tfs[stemmer.stem(token.lower())] += 1
        # lemmatize words
        # compare this to the result of stemming using the PorterStemmer
        tfs[pp.lemmatize(token.lower())] += 1
    return tfs


# Compute document frequencies for each term
def compute_doc_freqs(doc_term_freq):
    dfs = Counter()
    for tfs in doc_term_freq.values():
        for term in tfs.keys():
            dfs[term] += 1
    return dfs


# Build the vsm inverted index for each Wiki article
def train(doc):
    vsm_inverted_index.clear()
    # use the above two functions: extract_term_freqs & compute_doc_freqs
    # process the training data into term frequencies and document frequencies
    doc_term_freqs = {}
    sen_id = 0
    for sentence in doc:
        term_freqs = extract_term_freqs(sentence)
        doc_term_freqs[sen_id] = term_freqs
        sen_id += 1
    M = len(doc_term_freqs)
    doc_freqs = compute_doc_freqs(doc_term_freqs)
    
    # create the vsm inverted index
    for senid, term_freqs in doc_term_freqs.items():
        N = sum(term_freqs.values())
        length = 0
    
        # find tf*idf values and accumulate sum of squares 
        tfidf_values = []
        for term, count in term_freqs.items():
            tfidf = float(count) / N * math.log(M / float(doc_freqs[term]))
            tfidf_values.append((term, tfidf))
            length += tfidf ** 2

        # normalise documents by length and insert into index
        length = length ** 0.5
        for term, tfidf in tfidf_values:
            # note the inversion of the indexing, to be term -> (doc_id, score)
            vsm_inverted_index[term].append([senid, tfidf / length])
        
    # ensure posting lists are in sorted order
    for term, senids in vsm_inverted_index.items():
        senids.sort()


# Query the vsm inverted index
def query_vsm(query, k=1):
    accumulator = Counter()
    for term in query:
        postings = vsm_inverted_index[pp.lemmatize(term.lower())]
        for senid, weight in postings:
            accumulator[senid] += weight
    accumulator_list = sorted(accumulator.items(), key=lambda item: item[1], reverse=True)
    return accumulator_list
    # return accumulator.most_common(k)

