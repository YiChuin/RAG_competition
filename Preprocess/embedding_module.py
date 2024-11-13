import torch
import os
import json
# 分割文本
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
url = "dataset/preliminary"

class Document:
    def __init__(self, page_content, ids, category='insurance'):
        self.page_content = page_content
        self.metadata = {'ids':ids,'category':category}

def faq_change(value):
    context = ''
    for i in value:
        context += f"問題:'{i['question']}',答案:{i['answers']};"
    return context

def data_load(chunk_size=0, chunk_overlap=100):
    # 讀取 JSON 文件
    with open(url+'/finance_OCR(people).json', 'rb') as f:
        data_finance = json.load(f)

    with open(url+'/insurance_data.json', 'rb') as f:
        data_insurance = json.load(f)

    with open(url+'/pid_map_content.json', 'rb') as f:
        data_faq = json.load(f)

    documents_finance = [Document(value, key, category='finance') for key, value in data_finance.items()]
    documents_insurance = [Document(value, key, category='insurance') for key, value in data_insurance.items()]
    documents_faq = [Document(faq_change(value), key, category='faq') for key, value in data_faq.items()] # 變化FAQ的格式
    if chunk_size>0 :
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n"], chunk_size=chunk_size, chunk_overlap=chunk_overlap
                              )
        documents_finance = text_splitter.split_documents(documents_finance)
        documents_insurance = text_splitter.split_documents(documents_insurance)
        documents_faq = text_splitter.split_documents(documents_faq)

    return documents_finance, documents_insurance, documents_faq

def embedding(documents, category="insurance", batch_size=12, embedding_model='None', file_name=""):
    count = 0
    for i in range(0, len(documents), batch_size):
        if i+batch_size > len(documents):
            batch_size = len(documents)-i
        with torch.no_grad():
            batch = documents[i:i+batch_size]
            vectordb = Chroma.from_documents(
                documents=batch,
                embedding=embedding_model,
                persist_directory=os.path.join(url, f"chroma/{file_name}/db_{category}")
            )
            # vectordb.persist()
            count+=batch_size
            # 清理GPU空間
            torch.cuda.empty_cache()
            print(f'start embedding {category}:')
            print(f"-- no.{i+batch_size}/{len(documents)} --embedding ok!")
            vectordb = None
        print(f"len:{len(documents)},count:{count}")
        print(f"{category} embedding ok!")