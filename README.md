GM 美麗新世界
---

- dataset
  - 儲存不同類型資料的原始檔、OCR後的檔案
- Preprocess
  - 把dataset的資料轉成json .py
  - 把json 轉成 chroma (向量資料庫)
- Model
  - 檢索模型 :
    - 可輸出最相關文本名稱
    - 可輸出參考資料的文本。
  - LLM 模型
    -  待更新

--- 

Installation
---

### Using pip:

下載

    !pip install -U FlagEmbedding
    !pip install langchain
    !pip install langchain-chroma
    !pip install langchain_huggingface
