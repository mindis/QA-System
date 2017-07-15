import ner
import util
import bm25_ir as pir
import answer_ranking as ar
import pre_process as pp


def main():
    data = util.load_json('DataSets/QA_test.json')
    answers = []
    index = 0
    for wiki in data:
        sentences = wiki['sentences']
        # ner.extract_all(pp.preprocess(sentences, True))
        ner.extract_all(pp.sen_tokenize(sentences))
        pir.bm25(sentences, k1=0.8, b=0.5)
        questions = util.get_questions(wiki)
        for question in questions:
            answer_sent_ids = pir.query_bm25(question, k3=0)
            # print(question)
            if answer_sent_ids:
                answer = ar.answer_ranking(sentences, answer_sent_ids, question)
                answers.append(answer)
                # print(answer)
            else:
                answers.append('not sure')
        print("wiki:" + str(index))
        index += 1

    write_answer(answers)


def find_answer(answer_sent_id, question):
    entities = ner.get_entities(answer_sent_id)
    ques_type = ner.question_type(question)
    ranked = []
    for entity in entities:
        for k, v in entity.items():
            if k in question:
                continue
            else:
                if v == ques_type:
                    ranked.append(k)
                else:
                    continue
    if len(ranked) != 0:
        return ar.closer_to_open(question, ranked)
    else:
        return "not sure"


def write_answer(result):
    for i in range(len(result)):
        result[i] = result[i].replace(',', '-COMMA-')
        result[i] = result[i].replace('"', '')
    with open('result.csv', 'w', encoding='utf-8') as file:
        id = 1
        file.write('id' + ',' + 'answer\n')
        for answer in result:
            file.write(str(id) + ',' + answer.strip() + '\n')
            id += 1


if __name__ == "__main__":
    main()

