import ner
import util
import nltk
import pre_process as pp


# The answer ranking combining the three rules
def answer_ranking(sentences, answer_sent_ids, query):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    sentence = ''
    entities = []
    answer = ''
    sen_id = 0
    for id in answer_sent_ids:
        entities = ner.get_entities(id[0])
        label = 0
        if entities:
            for entity in entities:
                for k, v in entity.items():
                    if k.lower() not in stopwords:
                        label = 1
            if label == 1:
                sen_id = id[0]
                sentence = sentences[sen_id]
                break
    # print(sentence)
    # print(entities)
    ques_type = ner.question_type(query)
    # print(ques_type)
    tagged_sent = ner.tagged_sents[sen_id]
    entity_dict = ner.get_trans_entities(tagged_sent)
    entity_score = {}
    score_3 = closer_to_open(sentence, sen_id, query)
    for entity in entities:
        for k, v in entity.items():
            entity_score[k] = 0
    for entity in entities:
        for k, v in entity.items():
            if entity_score[k] == 0:
                entity_score[k] += rule_one(k, query)
                entity_score[k] += score_3[k]
                if v == ques_type:
                    entity_score[k] += 1
    value = sorted(entity_score.values())
    # print(entity_score)
    if entity_score != {}:
        for k, v in entity_score.items():
            if v == value[len(entity_score) - 1]:
                answer = k
        return answer
    else:
        return "not sure"


# Rule 1
# Answers whose content words all appear in the question should be ranked lowest
def rule_one(entity, query):
    score = 0
    entity = pp.lemmatize(entity.lower())
    if entity not in pp.stopwords and entity not in query:
        score = 1
    else:
        score = 0
    return score


# Rule 2
# Answers which match the question type should be ranked higher than those that don't
def match_question_type(entities, question):
    question_type = ner.question_type(question)
    score = 0
    for entity in entities:
        for x, y in entity.items():
            if question_type == y:
                # if question_type == 'OTHER':
                    # score = 0.5
                # else:
                    score = 1
    return score


# Rule 3
# Among entities of the same type,
# the preferred entity should be the one which is closer in the sentence to an open-class word from the question
def find_open_class_words(sentence):
    open_class_words = {}
    for i in range(len(sentence)):
        if sentence[i].lower() not in pp.stopwords and sentence[i] not in pp.punctuations:
                open_class_words[sentence[i]] = i
    return open_class_words


# Input is the question and entities, output is the entity score
def closer_to_open(sentence, answer_sent_id, query):
    entities = ner.get_entities(answer_sent_id)
    sen = find_open_class_words(sentence.split())
    entity_score = {}
    total_score = 0
    avg_score = 0
    for entity in entities:
        for k,v in entity.items():
            entity_score[k] = 0
    for entity in entities:
        for k, v in entity.items():
            for l, x in sen.items():
                if pp.lemmatize(l) in query:
                    if k.split()[0] in sen.keys():
                        entity_score[k] += abs(sen[k.split()[0]] - x)
    for k, v in entity_score.items():
        total_score += v
    if len(entities) != 0:
        avg_score = total_score / len(entities)
    for k, v in entity_score.items():
        entity_score[k] = util.sigmoid(avg_score - entity_score[k])
    return entity_score

