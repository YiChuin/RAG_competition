# RAG
<a href="https://github.com/YiChuin/RAG_competition/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=YiChuin/RAG_competition" />
</a>

Made with [contrib.rocks](https://contrib.rocks).
# 專案說明

## 目錄結構圖
請在Chuxin-Embedding-v2/內新增db_finance和db_insurance，連結在Chroma資料夾說明處
```
RAG_competition
├── Chroma/
│   ├── Chuxin-Embedding-v2/
│   │   └── db_faq/
├── Model/
│   ├── retrieval.py
│   └── retrieval_module.py
├── Preprocess/
│   ├── embedding.py
│   └── embedding_module.py
├── dataset/
│   ├── faq/
│   ├── finance/
│   ├── insurance/
│   ├── Test Dataset_Preliminary 1/
│   ├── preliminary/
│   │   ├── finance_OCR(people).json
│   │   ├── insurance_data.json
│   │   └── pid_map_content.json
├── output/
├── README.md
└── requirements.txt
```

---
- 資源配置: 使用Google Colab上的運算資源去做程式的運行，GPU為記憶體15GB的T4，Python版本為3.10.12

## 目錄說明
- dataset
  - faq、finance、insurance : 儲存不同類型資料的原始檔
  - Test Dataset_Preliminary 1 : 儲存900題問題和輸出範例的json檔
  - preliminary:儲存轉為json檔後的問題和資料
    - finance_OCR(people).json : 將finance資料夾內的檔案內容提取，並且對部分含圖片的檔案內容做人工OCR後存成的json檔 (因使用其他OCR模組的提取效果不佳)
    - insurance_data.json : 將insurance資料夾內的檔案內容提取出來做彙整並人工處理部分無關內容的json檔
    - pid_map_content.json : 將faq資料夾內的檔案內容提取出來的json檔
- Chroma
  - 儲存三種類型的資料做embedding後的資料
  - 由於db_finance和df_insurance大於上傳限制，請至以下連結下載並儲存進Chroma目錄內
  - 資料連結:  https://drive.google.com/drive/folders/103gOBxwf8gnJEqsI6d8KhpW9Y2dL-xsl?usp=sharing

儲存結果應為

```
├── Chroma/
│   ├── Chuxin-Embedding-v2/
│   │   └── db_faq/
│   │   └──  db_finance/
│   │   └──  db_insurance/
```

  
- output
  - 儲存各種embedding方式和切塊策略的預測結果

- Preprocess
  - 前處理程式碼，將finance_OCR(people).json、insurance_data.json、pid_map_content.json檔做切chunk和embedding，並存進Chroma (向量資料庫)內
    
- Model
  - 檢索模型 :
    - 可輸出最相關文本名稱
    - 可輸出參考資料的文本
    - 使用方式請見檔案夾內的RREADME

  - LLM 模型
    -  待更新
## 使用說明
1. 將資料從drive連結內存取到chroma資料夾內(db_faq直接覆蓋即可)
2. 載入requirements.txt的套件
   ```bash
   pip install -r requirements.txt
   ```
3. 檢查套件及python版本
   ```bash
   pip list
   python --version
   ```
   (執行以下前請檢查path是否正確，且終端機導到RAG_competition-main即可)
4. 執行 Preprocess內的embedding.py (可忽略，因reference資料已轉為向量儲存)
5. 執行Model內的retrieval.py(預測)
6. 檢查存在output_path內的預測結果
   
--- 

## Installation
---

## Using pip:

下載

    !pip install chromadb==0.5.18
    !pip install -U FlagEmbedding==1.3.2
    !pip install langchain==0.3.7
    !pip install langchain-chroma==0.1.4
    !pip install langchain_huggingface==0.1.2

