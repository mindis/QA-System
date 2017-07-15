import util
import ner
import basic_ir as ir
import bm25_ir as pir
import pre_process as pp
import answer_ranking as ar


"""
Answer sentence accuracy

# QA_train
basic ir -> 0.601633432631594
bm25 -> 0.66514631052324

# QA_dev
basic ir -> 0.5685926976249557
bm25 -> 0.641616448068061

* parameters for bm25
k1 = 0.8; b = 0.5; k3 = 0
"""


def evaluate_ir():
    data = util.load_json('DataSets/QA_train.json')
    total_ques_num = 0
    total_cor_num = 0

    for wiki in data:
        sentences = wiki['sentences']
        question_num = 0
        correct_answer = 0
        # ir.train(sentences)
        pir.bm25(sentences, k1=0.8, b=0.5)
        questions = util.get_questions(wiki)
        for question in questions:
            # answer_sent_id = ir.query_vsm(question)
            answer_sent_id = pir.query_bm25(question, k3=0)
            if answer_sent_id and answer_sent_id[0][0] == wiki['qa'][question_num]['answer_sentence']:
                correct_answer += 1
            question_num += 1
        total_ques_num += question_num
        total_cor_num += correct_answer
    return total_cor_num/total_ques_num

print(evaluate_ir())


def eva_train():
    data = util.load_json('DataSets/QA_dev.json')
    correct_answers = 0
    question_count = 0
    for wiki in data:
        index = 0
        sentences = wiki['sentences']
        ner.extract_all(pp.sen_tokenize(sentences))
        pir.bm25(sentences, k1=0.8, b=0.5)
        questions = util.get_questions(wiki)
        for question in questions:
            answer_sent_ids = pir.query_bm25(question, k3=0)
            if answer_sent_ids:
                answer = ar.answer_ranking(sentences, answer_sent_ids, question)
                correct_answer = wiki['qa'][index]['answer']
                if answer == correct_answer:
                    correct_answers += 1
            question_count += 1
            index += 1
    print('The number of correct answers: ', correct_answers)
    print('The number of total questions: ', question_count)
    print('Precision: ', correct_answers / question_count)

