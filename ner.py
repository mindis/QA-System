from nltk.tag import StanfordNERTagger
from util import isnumber
import os


cache_file_path = "entities.json"

NER_dir = os.path.join('NER_Tagger')
model_name = os.path.join(NER_dir, 'english.all.3class.distsim.crf.ser.gz')
jar_name = os.path.join(NER_dir, 'stanford-ner.jar')
st = StanfordNERTagger(model_filename=model_name, path_to_jar=jar_name)

tagged_sents = []


def extract_all(sentences):
    global tagged_sents
    tagged_sents = st.tag_sents(sentences)


def get_trans_entities(tagged_sent):
    entity_dict = []
    for entity in tagged_sent:
        entity_type = {entity[0]: entity[1]}
        if entity[1] == 'O':
            if isnumber(entity[0]):
                entity_type[entity[0]] = "NUMBER"
                entity_dict.append(entity_type)
                continue
            if entity[0].isalpha() and not entity[0].islower() and entity != tagged_sent[0]:
                entity_type[entity[0]] = "OTHER"
                entity_dict.append(entity_type)
                continue
        if entity[1] == 'ORGANIZATION':
            entity_type[entity[0]] = "OTHER"
            entity_dict.append(entity_type)
            continue
        entity_type[entity[0]] = entity[1]
        entity_dict.append(entity_type)
    return entity_dict


def get_merged_entities(entities):
    merged_entities = []
    start_type = ''
    if len(entities) != 0:
        for v in entities[0].values():
            start_type = v
        start_name = ''
        added = []
        for entity in entities:
            for k, v in entity.items():
                if v == start_type:
                    start_name = start_name + ' ' + k
                else:
                    if start_type != 'O':
                        merged_entities.append({start_name: start_type})
                        added.append(start_name)
                    start_type = v
                    start_name = k
        if start_name not in added and start_type != 'O':
            merged_entities.append({start_name: start_type})
        if not merged_entities and start_type != 'O':
            merged_entities.append({start_name: start_type})
    return merged_entities


def get_entities(sentence_id):
    tagged_sent = tagged_sents[sentence_id]
    entity_dict = get_trans_entities(tagged_sent)
    return get_merged_entities(entity_dict)


def question_type(question):
    weights = [0, 0, 0, 0]  # per, loc, num, other
    type = 'OTHER'
    num_kw = {'many', 'much', 'amount', 'length', 'time', 'day', 'year', 'decade', 'decades', 'long', 'old',
              'range', 'percent', 'level', 'average', 'population', 'wavelength'}
    per_kw = {'who', 'who\'s', 'whom', 'whose', 'person', 'name', 'president'}
    loc_kw = {'where', 'place', 'district', 'city', 'country'}
    # wh_words = {'what','where','who','whom'}
    # pre-processing questions
    # assign weight for each type depending on the keywords

    for word in question:
        word = str.lower(word)
        if word in per_kw:
            weights[0] += len(question)
        if word in loc_kw:
            weights[1] += len(question)
        if word in num_kw:
            weights[2] += len(question)
        else:
            weights[3] += 1
    index = weights.index(max(weights))
    if index == 0:
        type = 'PERSON'
    if index == 1:
        type = 'LOCATION'
    if index == 2:
        type = 'NUMBER'
    if index == 3:
        type = 'OTHER'
    # print(type)
    return type

