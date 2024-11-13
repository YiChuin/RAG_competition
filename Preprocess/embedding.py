## 載入模型 ##
import embedding_module as em
from langchain_huggingface import HuggingFaceEmbeddings

model_name = "chuxin-llm/Chuxin-Embedding"
model = HuggingFaceEmbeddings(model_name=model_name)
file_name = "Chuxin-Embedding-v2"

###### 載入data ######
## default ## chunk_size=0, chunk_overlap=100 ###### 
##   圖文        文字        json
documents_finance, documents_insurance, documents_faq = em.data_load(chunk_size=500, chunk_overlap=100)

###### 資料 to 向量 ######
## default ## documents, category="insurance", batch_size=12, embedding_model='None', file_name='' ######
em.embedding(documents_finance, category="finance", batch_size=6, embedding_model=model, file_name=file_name)
em.embedding(documents_insurance, category="insurance", batch_size=12, embedding_model=model, file_name=file_name)
em.embedding(documents_faq, category="faq", batch_size=12 ,embedding_model=model, file_name=file_name)
