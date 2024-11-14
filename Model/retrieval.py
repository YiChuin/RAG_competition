from FlagEmbedding import FlagReranker
from langchain_huggingface import HuggingFaceEmbeddings
import retrieval_module as re
import torch
import os

## import 模型 ##
# reranker_model = 'BAAI/bge-reranker-v2-m3'
reranker_model = "BAAI/bge-reranker-large"
reranker = FlagReranker(reranker_model, use_fp16=True, padding=True)

model_name = "chuxin-llm/Chuxin-Embedding"
model = HuggingFaceEmbeddings(model_name=model_name)
file_name = 'Chuxin-Embedding-v2'

## 檢索 ##
K = 30  #從資料庫中 選取前"幾"相關做輸出後，丟入reranking，自動輸出最相關的文本檔案名稱
url = "dataset/Test_Dataset_Preliminary_1"
question_path = os.path.join(url, 'questions_preliminary.json') 
output_path=  f'output/{file_name}/pred_retrieve_K{K}_{file_name}.json'
with torch.no_grad():
    re.retriever2json( k=K, output_path=output_path, model=model, file_name=file_name,
                      reranker = reranker, question_path = question_path)
    torch.cuda.empty_cache()

## 計算準確度 ##
# 答案
ground_truths_path = None
pred_path = output_path
if os.path.exists(pred_path) and ground_truths_path != None :
    re.cal(pred_path, ground_truths_path)
