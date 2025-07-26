# Check Worthiness Estimetion of Text Data

This research addresses the growing need to identify claims worth fact-checking, especially in the age of widespread misinformation. Focusing on English-language datasets from U.S. presidential debate transcripts, we apply a range of resampling methods to tackle data imbalance and explore multiple machine learning approaches from traditional models to fine-tuned LLMs using memory-efficient techniques like QLoRA.

## Key contributions include :

- How to handle huge ammount of imbalance text data.
  
- Prompt engineering (few shot, zero shot) for data labeling and data processing.

- Tweet data scrapping for validate the model, prompt engineering for label automation.
  
- Laveraging Data pruning techniques with LLM prompt engineering and  varies NLP libraries.

- Evaluation of linguistic, contextual, and semantic features.

- Ensemble strategies that significantly boost performance.

- Benchmarking on CLEF 2024's CheckThat! dataset and additional tweet data.

- Preliminary results show ensemble-based classical models outperform current CLEF 2024 LLM-based baselines.

- Fine-tuning encoder, decoder state of the art LLM Model with [QLoRA](https://huggingface.co/docs/peft/main/en/developer_guides/quantization) also give better performance compared to LoRA.
  
<!-- - Using both structured and unstructured data sources, create [retrieval-augmented LLM pipelines](https://github.com/monochandan/RAG-LLM-pipeline) for claim checkworthiness development. which will highlight the value of the RAG system in determining check worthiness. -->


With the rapid rise of online misinformation, it's increasingly important to prioritize which claims are worth fact-checking. Check-worthiness estimation tackles this by classifying whether a statement like a tweet or debate quote-merits verification. However, challenges such as subjectivity, data imbalance, and linguistic ambiguity make this task difficult.

While recent benchmarks like CheckThat! Lab at CLEF 2024 have seen dominance from transformer-based and LLM-based models (e.g., RoBERTa, GPT-4, LLaMA2), these often demand high computational resources. Whether using transformer-based models or traditional machine learning approaches, my focus was on efficiently addressing this problem in a trustworthy manner, while keeping in mind the variety and complexity of textual data. This project explores different approachs of ensemble-based traditional ML models, supported by resampling techniques, can remain competitive. I have conduct  experiment with [QLoRA](https://arxiv.org/abs/2305.14314) for memory-efficient fine-tuning of large models, offering a practical alternative to resource-heavy approaches.


## Used LLM models till now:
- gemini-1.5-flash - prompt engineering for class label automation.
- falcon-7b - data pruning by using prompt engineering.
- BERT - LLM model for text classification.
- phi-2 - LLM model for text classification.
- mistral-7b -  LLM model for text classification. (working)

## Classical model used till now (Hyperparameter Tuned):
- Random Forest
- XGB
- Decision Tree
- KNN
- GDB
- LGBM
- ADB

  .markdown-body .rdmd-table {
  --table-text: black;
  --table-head: #5b1c9f;
  --table-head-text: white;
  --table-stripe: #f0eaf7;
  --table-edges: rgba(34, 5, 64, 0.5);
  --table-row: white;
}

### 🔀 Voting Classifier (7 Models: RF, XGB, DT, KNN, GDB, LGBM, ADB)

| Metric     | Value |
|------------|-------|
| Accuracy   | 0.827 |
| Precision  | 0.821 |
| Recall     | 0.827 |
| F1 Score   | 0.823 |
  
### 🔀 Voting Classifier (3 Models: GDB, LGBM, ADB)

| Metric     | Value |
|------------|-------|
| Accuracy   | 0.809 |
| Precision  | 0.806 |
| Recall     | 0.809 |
| F1 Score   | 0.808 |

### 🔀 Voting Classifier (5 Models: DT, KNN, RF, XGB, ADB)

| Metric     | Value |
|------------|-------|
| Accuracy   | 0.801 |
| Precision  | 0.793 |
| Recall     | 0.801 |
| F1 Score   | 0.796 |

### Blending Classifier (Working)

### Stacking Classifier (Working)

### 🌲 Random Forest Classifier (RFC)

| Metric     | Value |
|------------|-------|
| Accuracy   | 0.745 |
| Precision  | 0.728 |
| Recall     | 0.745 |
| F1 Score   | 0.734 |

### 🌳 Decision Tree Classifier (DT)

| Metric     | Value |
|------------|-------|
| Accuracy   | 0.721 |
| Precision  | 0.729 |
| Recall     | 0.721 |
| F1 Score   | 0.725 |

### 🤖 K-Nearest Neighbors (KNN)

| Metric     | Value |
|------------|-------|
| Accuracy   | 0.674 |
| Precision  | 0.666 |
| Recall     | 0.674 |
| F1 Score   | 0.670 |

### 🚀 XGBoost Classifier (XGB)

| Metric     | Value |
|------------|-------|
| Accuracy   | 0.795 |
| Precision  | 0.805 |
| Recall     | 0.795 |
| F1 Score   | 0.799 |

### Parameter Efficient Fine Tuning (PEFT) QLoRA, Model: Microsoft BERT (Encoder)

### Evaluation Dataset Scores

| Metric     | Value        |
|------------|--------------|
| Accuracy   | 0.9176       |
| F1 Score   | 0.8399       |
| Precision  | 0.7611       |
| Recall     | 0.9370       |

## This result has already surpassed the best scores from CLEF 2024, as seen [here](https://checkthat.gitlab.io/clef2024/task1/).

[credit](https://checkthat.gitlab.io/clef2024/task1/)
<img width="506" alt="image" src="https://github.com/user-attachments/assets/e5e0c9f1-8f0a-485d-9d55-04adc89a4833" />

