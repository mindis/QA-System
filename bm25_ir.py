import math
from collections import defaultdict, Counter
import pre_process as pp


'''
BM25 scoring formula for VSM IR
Parameters k1, b, k3 need to be tuned
Defaults k1 = 1.5, b = 0.5, k3 = 0

BM25 gives better results than
basic vsm in basic_ir.py
'''

bm25_index = defaultdict(list)


# Collect term frequencies for each sentence in document
def extract_term_freqs(sen):
    # bag-of-words representation
    tfs = Counter()
    sentence = pp.process_sen(sen, True)
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


def bm25(doc, k1, b):
    bm25_index.clear()
    # use the above two functions: extract_term_freqs & compute_doc_freqs
    # process the training data into term frequencies and document frequencies
    doc_term_freqs = {}
    sen_id = 0
    sen_len_sum = 0
    for sentence in doc:
        term_freqs = extract_term_freqs(sentence)
        doc_term_freqs[sen_id] = term_freqs
        sen_id += 1
        sen_len_sum += sum(term_freqs.values())
    M = len(doc_term_freqs)
    avg_sen_length = float(sen_len_sum) / M
    doc_freqs = compute_doc_freqs(doc_term_freqs)

    # create the bm25 vsm
    for senid, term_freqs in doc_term_freqs.items():
        N = sum(term_freqs.values())
        term_weighting_values = []
        for term, count in term_freqs.items():
            # 1. find idf values
            idf = math.log((M - float(doc_freqs[term]) + 0.5) / (float(doc_freqs[term]) + 0.5))

            # 2. find tf and doc length values
            K = k1 * (1 - b + b * (float(N) / avg_sen_length)) + count
            tf_doc_len = float(k1 + 1) * count / K

            score = idf * tf_doc_len
            term_weighting_values.append((term, score))

        for term, score in term_weighting_values:
            # note the inversion of the indexing, to be term -> (doc_id, score)
            bm25_index[term].append([senid, score])

    # ensure posting lists are in sorted order
    for term, senids in bm25_index.items():
        senids.sort()


# Query the bm25 index
def query_bm25(query, k3, k=1):
    # process query into term frequencies
    qfs = Counter()
    for token in query:
        qfs[pp.lemmatize(token.lower())] += 1

    accumulator = Counter()
    for term in query:
        term = pp.lemmatize(term.lower())
        # 3. find query tf values
        query_tf = float((k3 + 1) * qfs[term]) / (k3 + qfs[term])
        postings = bm25_index[term]
        for senid, weight in postings:
            accumulator[senid] += weight * query_tf
    accumulator_list = sorted(accumulator.items(), key=lambda item: item[1], reverse=True)
    return accumulator_list
    # return accumulator.most_common(k)

