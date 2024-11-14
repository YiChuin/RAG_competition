import json
import os
from langchain_chroma import Chroma
url = "dataset/preliminary"
#　計算輸出的json & ground_truths 的準確度
def cal(pred_path, ground_truths_path = os.path.join(url, 'ground_truths_example.json')):
    with open(ground_truths_path, 'r') as f:
        ground_truth = json.load(f)['ground_truths']

    pred_retrieve = pred_path
    with open(pred_retrieve, 'r') as f:
        predictions = json.load(f)['answers']

  # 題數
    total_count = len(ground_truth)
    total_finance = 0
    total_insurance = 0
    total_faq = 0
    # 答對的題數
    correct_count = 0
    correct_finance = 0
    correct_insurance = 0
    correct_faq = 0
    # 答錯的題號
    error_finance = []
    error_insurance = []
    error_faq = []

    for gt, pred in zip(ground_truth, predictions):
        if gt['category'] == "finance":
            total_finance += 1
            if gt['qid'] == pred['qid'] and str(gt['retrieve']) in pred['retrieve']:
                correct_count += 1
                correct_finance +=1
            else:
                error_finance.append(gt['qid'])
        elif gt['category'] == "insurance":
            total_insurance += 1
            if gt['qid'] == pred['qid'] and str(gt['retrieve']) in pred['retrieve']:
                correct_count += 1
                correct_insurance +=1
            else:
                error_insurance.append(gt['qid'])
        elif gt['category'] == "faq":
            total_faq += 1
            if gt['qid'] == pred['qid'] and str(gt['retrieve']) in pred['retrieve']:
                correct_count += 1
                correct_faq +=1
            else:
                error_faq.append(gt['qid'])

    accuracy = correct_count / total_count
    accuracy_finance = correct_finance / total_finance
    accuracy_insurance = correct_insurance / total_insurance
    accuracy_faq = correct_faq / total_faq
    print(f" 準確度:{accuracy} \n 答對題數:{correct_count}\n 總題數:{total_count}\n")
    print(f" finance的準確度:{accuracy_finance} \n 答對題數:{correct_finance}\n 總題數:{total_finance}\n 答錯項:{error_finance}\n")
    print(f" insurance的準確度:{accuracy_insurance} \n 答對題數:{correct_insurance}\n 總題數:{total_insurance}\n 答錯項:{error_insurance}\n")
    print(f" faq的準確度:{accuracy_faq} \n 答對題數:{correct_faq}\n 總題數:{total_faq}\n 答錯項:{error_faq}\n")


def find_top_k_points(results, scores, K=1):
    # 將分數和 id 配對成 tuple 的 list
    id_score_pairs = [(result.metadata['ids'], score, result.page_content) for result, score in zip(results, scores)]

    # 按分數排序，從高到低
    id_score_pairs.sort(key=lambda x: x[1], reverse=True)

    # 取得前 K 個分數最高的 (id, score) 並返回 id 列表
    top_k_ids = [id for id, score, content in id_score_pairs[:K]]
    top_k_contents = [content for id, score, content in id_score_pairs[:K]]
    return top_k_ids, top_k_contents

def find_top_point(results, scores, K=1):
    id_score_pairs = [(result.metadata['ids'], score, result.page_content) for result, score in zip(results, scores)]
    id_score_pairs.sort(key=lambda x: x[1], reverse=True)
    top_k_ids = [id for id, score, content in id_score_pairs[:K]]
    return str(top_k_ids[0])


# 把檢索到的資訊輸出。 # output答案可能在哪個pdf
def retriever2json(k=10, output_path=os.path.join(url, 'pred_retrieve_K10.json'), 
    model=None, file_name='', reranker=None,
    question_path = os.path.join(url, 'questions_example.json') ):
    # 讀取問題.json
    with open(question_path, 'rb') as f:
        qs_ref = json.load(f)  # 讀取問題檔案

    # finance
    persist_finance = f'chroma/{file_name}/db_finance'
    vectordb_finance = Chroma(persist_directory=persist_finance, embedding_function=model)
    # insurance
    persist_insurance =  f'chroma/{file_name}/db_insurance'
    vectordb_insurance = Chroma(persist_directory=persist_insurance, embedding_function=model)
    # faq
    persist_faq =  f'chroma/{file_name}/db_faq'
    vectordb_faq = Chroma(persist_directory=persist_faq, embedding_function=model)

    answer_dict = {"answers": []}  # 初始化字典
    count = 0
    for q_dict in qs_ref['questions']:
        if q_dict['category'] == 'finance':
            # 進行檢索
            ids_ = [f'{i}.pdf' for i in q_dict['source']]  # '?.pdf'的版本
            results = vectordb_finance.similarity_search(q_dict['query'],k=k,filter={'ids': {"$in": ids_} })
            scores = reranker.compute_score([[q_dict['query'], result.page_content] for result in results])

            top_point_id = find_top_point(results, scores) # str
            
            # 將結果加入字典
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": top_point_id})

        elif q_dict['category'] == 'insurance':
            # 進行檢索
            ids_ = [f'{i}.pdf' for i in q_dict['source']]  # '?.pdf'的版本
            results = vectordb_insurance.similarity_search(q_dict['query'],k=k,filter={'ids': {"$in": ids_} })
            scores = reranker.compute_score([[q_dict['query'], result.page_content] for result in results])

            top_point_id = find_top_point(results, scores) # str
 
            # 將結果加入字典
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": top_point_id})

        elif q_dict['category'] == 'faq':
            # 進行檢索
            ids_ = [str(i) for i in q_dict['source']]
            results = vectordb_faq.similarity_search(q_dict['query'],k=k,filter={'ids': {"$in": ids_} })
            results_id = [i.metadata['ids'] for i in results] #list裡面為str
            if len(results_id) ==1 :
                top_point_id = str(results_id[0])
            # 將結果加入字典
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": top_point_id})

        else:
            raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤
        count+=1

    # 保存結果
    with open(output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符
    print('successful!')
    vectordb_finance=None
    vectordb_insurance=None
    vectordb_faq=None
    return None


# 把檢索到的資訊輸出。 # 輸出為對應的pdf名稱以及對應的切割文本
from langchain_huggingface import HuggingFaceEmbeddings
def retriever2json_content(k=30, out=1, output_path=os.path.join(url, 'pred_retrieve_K10.json'), 
    model=None, file_name='', reranker=None,
    question_path = os.path.join(url, 'questions_example.json') ):
    # 讀取問題.json
    with open(question_path, 'rb') as f:
        qs_ref = json.load(f)  # 讀取問題檔案

    # finance
    persist_finance = f'chroma/{file_name}/db_finance'
    vectordb_finance = Chroma(persist_directory=persist_finance, embedding_function=model)
    # insurance
    persist_insurance =  f'chroma/{file_name}/db_insurance'
    vectordb_insurance = Chroma(persist_directory=persist_insurance, embedding_function=model)
    # faq
    persist_faq =  f'chroma/{file_name}/db_faq'
    vectordb_faq = Chroma(persist_directory=persist_faq, embedding_function=model)

    answer_dict = {"answers": []}  # 初始化字典
    count = 0
    for q_dict in qs_ref['questions']:
        if q_dict['category'] == 'finance':
            # 進行檢索
            ids_ = [f'{i}.pdf' for i in q_dict['source']]  # '?.pdf'的版本
            results = vectordb_finance.similarity_search(q_dict['query'],k=k,filter={'ids': {"$in": ids_} })
            # reranking
            scores = reranker.compute_score([[q_dict['query'], result.page_content] for result in results])

            out_index, out_content = find_top_k_points(results, scores, K=out)
            out_index = [ i.split('.')[0] for i in out_index]
            # 將結果加入字典
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": out_index, "content": out_content})

        elif q_dict['category'] == 'insurance':
            # 進行檢索
            ids_ = [f'{i}.pdf' for i in q_dict['source']]  # '?.pdf'的版本
            results = vectordb_insurance.similarity_search(q_dict['query'],k=k,filter={'ids': {"$in": ids_} })
            # reranking
            scores = reranker.compute_score([[q_dict['query'], result.page_content] for result in results])

            out_index, out_content = find_top_k_points(results, scores, K=out)
            out_index = [ i.split('.')[0] for i in out_index]
            # 將結果加入字典
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": out_index, "content": out_content})

        elif q_dict['category'] == 'faq':
            # 進行檢索
            ids_ = [str(i) for i in q_dict['source']]
            results = vectordb_faq.similarity_search(q_dict['query'],k=k,filter={'ids': {"$in": ids_} })
            # reranking
            scores = reranker.compute_score([[q_dict['query'], result.page_content] for result in results])
            
            out_index, out_content = find_top_k_points(results, scores, K=out)
            out_index = [ i.split('.')[0] for i in out_index]
            # 將結果加入字典
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": out_index, "content": out_content})

        else:
            raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤
        count+=1

    # 保存結果
    with open(output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符
    print('successful!')
    vectordb_finance=None
    vectordb_insurance=None
    vectordb_faq=None
    return None