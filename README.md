# Check Worthiness Estimetion of Text Data

With the rapid rise of online misinformation, it's increasingly important to prioritize which claims are worth fact-checking. Check-worthiness estimation tackles this by classifying whether a statement like a tweet or debate quote-merits verification. However, challenges such as subjectivity, data imbalance, and linguistic ambiguity make this task difficult.

While recent benchmarks like CheckThat! Lab at [CLEF 2024](http://clef2024.clef-initiative.eu/index.php?page=Pages/lab_pages/checkthat.html) have seen dominance from transformer-based and LLM-based models (e.g., RoBERTa, GPT-4, LLaMA2), these often demand high computational resources. Whether using transformer-based models or traditional machine learning approaches, my focus was on efficiently addressing this problem in a trustworthy manner, while keeping in mind the variety and complexity of textual data. This project explores different approachs of ensemble-based traditional ML models, supported by resampling techniques, can remain competitive. I have conduct  experiment with [QLoRA](https://arxiv.org/abs/2305.14314) for memory-efficient fine-tuning of large models, offering a practical alternative to resource-heavy approaches.


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

- Fine-tuning state of the art LLM Model with [QLoRA](https://huggingface.co/docs/peft/main/en/developer_guides/quantization) also give better performance compared to LoRA.
  
<!-- - Using both structured and unstructured data sources, create [retrieval-augmented LLM pipelines](https://github.com/monochandan/RAG-LLM-pipeline) for claim checkworthiness development. which will highlight the value of the RAG system in determining check worthiness. -->

### Custom [Ensemble](https://github.com/monochandan/Text-CheckWorthy-Classification-Master-Thesis/blob/main/Ensembling_techniques/Ensemble.py) Learning
Implemented advanced **blending and stacking ensembles** using:
- Manual out-of-fold training and prediction logic
- Integration of diverse base models (e.g., XGBoost, Logistic Regression, Decision Tree, Random Forest, Ada Boost, Gradient Boosting, Light GBM, KNN)
- Final meta-learner trained on base model predictions
Achieved improved F1 scores compared to traditional ensemble methods (e.g., VotingClassifier).


## Used LLM models till now:
- gemini-1.5-flash - prompt engineering for class label automation.
- falcon-7b - data pruning by using prompt engineering.
- [BERT](https://github.com/monochandan/Text-CheckWorthy-Classification-Master-Thesis/blob/main/LLM/PEFT-BERT-QLoRa--got-result_1.ipynb) - LLM model for text classification.
- multilingual BERT
- xlm-RoBERTa
<!--- mistral-7b -  LLM model for text classification. (working)-->

## Classical model used till now Hyperparameter Tuned for [English](https://github.com/monochandan/Text-CheckWorthy-Classification-Master-Thesis/blob/main/hp-tuning/hp-tuning.ipynb), [Arabic](https://github.com/monochandan/Text-CheckWorthy-Classification-Master-Thesis/blob/main/hp-tuning/arabic_hp_tuning.ipynb) and [Spanish]()
- Random Forest
- XGB
- Decision Tree
- KNN
- GDB
- LGBM
- ADB

### BenchMark Dataset from [CLEF2024](https://gitlab.com/checkthat_lab/clef2024-checkthat-lab/-/tree/main/task1/data)
<table>
  <tr>
    <td><img width="1080" height="960" alt="Image" src="https://github.com/user-attachments/assets/fb19c0cd-8a72-4d6b-ba79-31c39aa6f934" /></td>
      <!--<td><img width="567" height="455" alt="Image" src="https://github.com/user-attachments/assets/36c503fd-e726-456e-89c8-a5269982c679" /></td>-->
  </tr>
</table>

## English

### 1. Voting Classifier (4 Models: 'dt', 'knn', 'rf', 'xgb')



<table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not Checkworthy</th>
    <th>Checkworthy</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Benchmark Data</td>
    <td align="center">17088</td>
    <td align="center">5413</td>
    <td align="center">0.934</td>
    <td align="center">0.934</td>
    <td align="center">0.934</td>
    <td align="center">0.931</td>
  </tr>
  <tr>
    <td>Resampled (Under-sampled) Data</td>
    <td align="center">9896</td>
    <td align="center">1254</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td>Tweet Data (e.g., Recent Political and Covid Pandemic Tweets)</td>
    <td align="center">172</td>
    <td align="center">59</td>
    <td align="center">0.895</td>
    <td align="center">0.898</td>
    <td align="center">0.895</td>
    <td align="center">0.896</td>
  </tr>
</table>
  



### 2. Voting Classifier (6 Models: 'rf', 'xgb', 'dt', 'knn', 'lgbm', 'adb')

<!--| Metric     | Value |
|------------|-------|
| Accuracy   | 0.801 |
| Precision  | 0.793 |
| Recall     | 0.801 |
| F1 Score   | 0.796 |-->


<table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not Checkworthy</th>
    <th>Checkworthy</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Benchmark Data</td>
    <td align="center">17088</td>
    <td align="center">5413</td>
    <td align="center">0.946</td>
    <td align="center">0.947</td>
    <td align="center">0.946</td>
    <td align="center">0.945</td>
  </tr>
  <tr>
    <td>Resampled (Under-sampled) Data</td>
   <td align="center">9896</td>
    <td align="center">1254</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td>Tweet Data (e.g., Recent Political and Covid Pandemic Tweets)</td>
    <td align="center">172</td>
    <td align="center">59</td>
    <td align="center">0.924</td>
    <td align="center">0.924</td>
    <td align="center">0.924</td>
    <td align="center">0.924</td>
  </tr>
</table>

### 3. Voting Classifier (3 Models: 'gdb', 'lgbm', 'adb')

<!--| Metric     | Value |
|------------|-------|
| Accuracy   | 0.801 |
| Precision  | 0.793 |
| Recall     | 0.801 |
| F1 Score   | 0.796 |-->


<table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not Checkworthy</th>
    <th>Checkworthy</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Benchmark Data</td>
    <td align="center">17088</td>
    <td align="center">5413</td>
    <td align="center">0.943</td>
    <td align="center">0.945</td>
    <td align="center">0.943</td>
    <td align="center">0.941</td>
  </tr>
  <tr>
    <td>Resampled (Under-sampled) Data</td>
   <td align="center">9896</td>
    <td align="center">1254</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td>Tweet Data (e.g., Recent Political and Covid Pandemic Tweets)</td>
    <td align="center">172</td>
    <td align="center">59</td>
    <td align="center">0.925</td>
    <td align="center">0.925</td>
    <td align="center">0.925</td>
    <td align="center">0.925</td>
  </tr>
</table>

### 4. Voting Classifier (8 Hyperparameter Tuned Models: 'dt', 'knn', 'rf', 'xgb', 'adb')

<!--| Metric     | Value |
|------------|-------|
| Accuracy   | 0.801 |
| Precision  | 0.793 |
| Recall     | 0.801 |
| F1 Score   | 0.796 |-->


<table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not Checkworthy</th>
    <th>Checkworthy</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Benchmark Data</td>
    <td align="center">17088</td>
    <td align="center">5413</td>
    <td align="center">0.934</td>
    <td align="center">0.936</td>
    <td align="center">0.934</td>
    <td align="center">0.931</td>
  </tr>
  <tr>
    <td>Resampled (Under-sampled) Data</td>
    <td align="center">9896</td>
    <td align="center">1254</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td>Tweet Data (e.g., Recent Political and Covid Pandemic Tweets)</td>
    <td align="center">172</td>
    <td align="center">59</td>
    <td align="center">0.895</td>
    <td align="center">0.898</td>
    <td align="center">0.895</td>
    <td align="center">0.896</td>
  </tr>
</table>

### 5. Voting Classifier (3 Models: 'rf', 'xgb', 'dt')

<!--| Metric     | Value |
|------------|-------|
| Accuracy   | 0.801 |
| Precision  | 0.793 |
| Recall     | 0.801 |
| F1 Score   | 0.796 |-->


<table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not Checkworthy</th>
    <th>Checkworthy</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Benchmark Data</td>
    <td align="center">17088</td>
    <td align="center">5413</td>
    <td align="center">0.953</td>
    <td align="center">0.953</td>
    <td align="center">0.953</td>
    <td align="center">0.953</td>
  </tr>
  <tr>
    <td>Resampled (Under-sampled) Data</td>
   <td align="center">9896</td>
    <td align="center">1254</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td>Tweet Data (e.g., Recent Political and Covid Pandemic Tweets)</td>
    <td align="center">172</td>
    <td align="center">59</td>
    <td align="center">0.914</td>
    <td align="center">0.919</td>
    <td align="center">0.914</td>
    <td align="center">0.916</td>
  </tr>
</table>

### 6. Voting Classifier (4 Models: 'xgb', 'lgbm', 'adb', 'lr')

<!--| Metric     | Value |
|------------|-------|
| Accuracy   | 0.801 |
| Precision  | 0.793 |
| Recall     | 0.801 |
| F1 Score   | 0.796 |-->


<table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not Checkworthy</th>
    <th>Checkworthy</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Benchmark Data</td>
    <td align="center">17088</td>
    <td align="center">5413</td>
    <td align="center">0.958</td>
    <td align="center">0.958</td>
    <td align="center">0.958</td>
    <td align="center">0.958</td>
  </tr>
  <tr>
    <td>Resampled (Under-sampled) Data</td>
   <td align="center">9896</td>
    <td align="center">1254</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td>Tweet Data (e.g., Recent Political and Covid Pandemic Tweets)</td>
    <td align="center">172</td>
    <td align="center">59</td>
    <td align="center">0.919</td>
    <td align="center">0.924</td>
    <td align="center">0.919</td>
    <td align="center">0.921</td>
  </tr>
</table>

### 7. Voting Classifier (8 Hyperparameter Tuned Models: 'dt', 'knn', 'rf', 'xgb', 'gdb', 'lgbm', 'adb', 'lr')

<!--| Metric     | Value |
|------------|-------|
| Accuracy   | 0.801 |
| Precision  | 0.793 |
| Recall     | 0.801 |
| F1 Score   | 0.796 |-->


<table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not Checkworthy</th>
    <th>Checkworthy</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Benchmark Data</td>
    <td align="center">17088</td>
    <td align="center">5413</td>
    <td align="center">0.957</td>
    <td align="center">0.957</td>
    <td align="center">0.957</td>
    <td align="center">0.956</td>
  </tr>
  <tr>
    <td>Resampled (Under-sampled) Data</td>
   <td align="center">9896</td>
    <td align="center">1254</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td>Tweet Data (e.g., Recent Political and Covid Pandemic Tweets)</td>
    <td align="center">172</td>
    <td align="center">59</td>
    <td align="center">0.926</td>
    <td align="center">0.928</td>
    <td align="center">0.926</td>
    <td align="center">0.927</td>
  </tr>
</table>

### Blending Classifier ('rf', 'xgb', 'dt', 'knn', 'lgbm', 'adb')

<table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not Checkworthy</th>
    <th>Checkworthy</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Benchmark Data</td>
    <td align="center">17088</td>
    <td align="center">5413</td>
    <td align="center">0.956</td>
    <td align="center">0.956</td>
    <td align="center">0.956</td>
    <td align="center">0.955</td>
  </tr>
  <tr>
    <td>Resampled (Under-sampled) Data</td>
    <td align="center">9896</td>
    <td align="center">1254</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td>Tweet Data (e.g., Recent Political and Covid Pandemic Tweets)</td>
    <td align="center">172</td>
    <td align="center">59</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
</table>



### Stacking Classifier (Working)

### Random Forest Classifier (RFC)

<table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not Checkworthy</th>
    <th>Checkworthy</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Benchmark Data</td>
    <td align="center">17088</td>
    <td align="center">5413</td>
    <td align="center">0.825</td>
    <td align="center">0.858</td>
    <td align="center">0.825</td>
    <td align="center">0.834</td>
  </tr>
  <tr>
    <td>Resampled (Under-sampled) Data</td>
    <td align="center">9896</td>
    <td align="center">1254</td>
    <td align="center">0.745</td>
    <td align="center">0.728</td>
    <td align="center">0.745</td>
    <td align="center">0.734</td>
  </tr>
  <tr>
    <td>Tweet Data (e.g., Recent Political and Covid Pandemic Tweets)</td>
    <td align="center">172</td>
    <td align="center">59</td>
    <td align="center">0.841</td>
    <td align="center">0.838</td>
    <td align="center">0.841</td>
    <td align="center">0.840</td>
  </tr>
</table>




### Decision Tree Classifier (DT)

<!--| Metric     | Value |
|------------|-------|
| Accuracy   | 0.721 |
| Precision  | 0.729 |
| Recall     | 0.721 |
| F1 Score   | 0.725 |-->

<table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not Checkworthy</th>
    <th>Checkworthy</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Benchmark Data</td>
    <td align="center">17088</td>
    <td align="center">5413</td>
    <td align="center">0.635</td>
    <td align="center">0.788</td>
    <td align="center">0.635</td>
    <td align="center">0.663</td>
  </tr>
  <tr>
    <td>Resampled (Under-sampled) Data</td>
    <td align="center">9896</td>
    <td align="center">1254</td>
    <td align="center">0.721</td>
    <td align="center">0.729</td>
    <td align="center">0.721</td>
    <td align="center">0.725</td>
  </tr>
  <tr>
    <td>Tweet Data (e.g., Recent Political and Covid Pandemic Tweets)</td>
    <td align="center">172</td>
    <td align="center">59</td>
    <td align="center">0.763</td>
    <td align="center">0.822</td>
    <td align="center">0.763</td>
    <td align="center">0.779</td>
  </tr>
</table>



### K-Nearest Neighbors (KNN)

<!--| Metric     | Value |
|------------|-------|
| Accuracy   | 0.674 |
| Precision  | 0.666 |
| Recall     | 0.674 |
| F1 Score   | 0.670 |-->

<table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not Checkworthy</th>
    <th>Checkworthy</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Benchmark Data</td>
    <td align="center">17088</td>
    <td align="center">5413</td>
    <td align="center">0.768</td>
    <td align="center">0.732</td>
    <td align="center">0.768</td>
    <td align="center">0.736</td>
  </tr>
  <tr>
    <td>Resampled (Under-sampled) Data</td>
    <td align="center">9896</td>
    <td align="center">1254</td>
    <td align="center">0.674</td>
    <td align="center">0.666</td>
    <td align="center">0.674</td>
    <td align="center">0.670</td>
  </tr>
  <tr>
    <td>Tweet Data (e.g., Recent Political and Covid Pandemic Tweets)</td>
    <td align="center">172</td>
    <td align="center">59</td>
    <td align="center">0.673</td>
    <td align="center">0.691</td>
    <td align="center">0.673</td>
    <td align="center">0.681</td>
  </tr>
</table>




### XGBoost Classifier (XGB)

<!--| Metric     | Value |
|------------|-------|
| Accuracy   | 0.795 |
| Precision  | 0.805 |
| Recall     | 0.795 |
| F1 Score   | 0.799 |-->

<table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not Checkworthy</th>
    <th>Checkworthy</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Benchmark Data</td>
    <td align="center">17088</td>
    <td align="center">5413</td>
    <td align="center">0.957</td>
    <td align="center">0.957</td>
    <td align="center">0.957</td>
    <td align="center">0.956</td>
  </tr>
  <tr>
    <td>Resampled (Under-sampled) Data</td>
    <td align="center">9896</td>
    <td align="center">1254</td>
    <td align="center">0.795</td>
    <td align="center">0.805</td>
    <td align="center">0.795</td>
    <td align="center">0.799</td>
  </tr>
  <tr>
    <td>Tweet Data (e.g., Recent Political and Covid Pandemic Tweets)</td>
    <td align="center">172</td>
    <td align="center">59</td>
    <td align="center">0.925</td>
    <td align="center">0.930</td>
    <td align="center">0.925</td>
    <td align="center">0.926</td>
  </tr>
</table>

### Gradient Boosting Classifier (GDB)

<!--| Metric     | Value |
|------------|-------|
| Accuracy   | 0.795 |
| Precision  | 0.805 |
| Recall     | 0.795 |
| F1 Score   | 0.799 |-->

<table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not Checkworthy</th>
    <th>Checkworthy</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Benchmark Data</td>
    <td align="center">17088</td>
    <td align="center">5413</td>
    <td align="center">0.957</td>
    <td align="center">0.957</td>
    <td align="center">0.957</td>
    <td align="center">0.956</td>
  </tr>
  <tr>
    <td>Resampled (Under-sampled) Data</td>
    <td align="center">9896</td>
    <td align="center">1254</td>
    <td align="center">0.795</td>
    <td align="center">0.805</td>
    <td align="center">0.795</td>
    <td align="center">0.799</td>
  </tr>
  <tr>
    <td>Tweet Data (e.g., Recent Political and Covid Pandemic Tweets)</td>
    <td align="center">172</td>
    <td align="center">59</td>
    <td align="center">0.910</td>
    <td align="center">0.911</td>
    <td align="center">0.910</td>
    <td align="center">0.911</td>
  </tr>
</table>

### Light GBM Classifier

<!--| Metric     | Value |
|------------|-------|
| Accuracy   | 0.795 |
| Precision  | 0.805 |
| Recall     | 0.795 |
| F1 Score   | 0.799 |-->

<table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not Checkworthy</th>
    <th>Checkworthy</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Benchmark Data</td>
    <td align="center">17088</td>
    <td align="center">5413</td>
    <td align="center">0.957</td>
    <td align="center">0.957</td>
    <td align="center">0.957</td>
    <td align="center">0.956</td>
  </tr>
  <tr>
    <td>Resampled (Under-sampled) Data</td>
    <td align="center">9896</td>
    <td align="center">1254</td>
    <td align="center">0.795</td>
    <td align="center">0.805</td>
    <td align="center">0.795</td>
    <td align="center">0.799</td>
  </tr>
  <tr>
    <td>Tweet Data (e.g., Recent Political and Covid Pandemic Tweets)</td>
    <td align="center">172</td>
    <td align="center">59</td>
    <td align="center">0.921</td>
    <td align="center">0.921</td>
    <td align="center">0.921</td>
    <td align="center">0.921</td>
  </tr>
</table>

### Ada Boost Classifier

<!--| Metric     | Value |
|------------|-------|
| Accuracy   | 0.795 |
| Precision  | 0.805 |
| Recall     | 0.795 |
| F1 Score   | 0.799 |-->

<table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not Checkworthy</th>
    <th>Checkworthy</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Benchmark Data</td>
    <td align="center">17088</td>
    <td align="center">5413</td>
    <td align="center">0.957</td>
    <td align="center">0.957</td>
    <td align="center">0.957</td>
    <td align="center">0.956</td>
  </tr>
  <tr>
    <td>Resampled (Under-sampled) Data</td>
    <td align="center">9896</td>
    <td align="center">1254</td>
    <td align="center">0.795</td>
    <td align="center">0.805</td>
    <td align="center">0.795</td>
    <td align="center">0.799</td>
  </tr>
  <tr>
    <td>Tweet Data (e.g., Recent Political and Covid Pandemic Tweets)</td>
    <td align="center">172</td>
    <td align="center">59</td>
    <td align="center">0.906</td>
    <td align="center">0.904</td>
    <td align="center">0.906</td>
    <td align="center">0.904</td>
  </tr>
</table>

### Logistic Regression

<!--| Metric     | Value |
|------------|-------|
| Accuracy   | 0.795 |
| Precision  | 0.805 |
| Recall     | 0.795 |
| F1 Score   | 0.799 |-->

<table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not Checkworthy</th>
    <th>Checkworthy</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Benchmark Data</td>
    <td align="center">17088</td>
    <td align="center">5413</td>
    <td align="center">0.957</td>
    <td align="center">0.957</td>
    <td align="center">0.957</td>
    <td align="center">0.956</td>
  </tr>
  <tr>
    <td>Resampled (Under-sampled) Data</td>
    <td align="center">9896</td>
    <td align="center">1254</td>
    <td align="center">0.795</td>
    <td align="center">0.805</td>
    <td align="center">0.795</td>
    <td align="center">0.799</td>
  </tr>
  <tr>
    <td>Tweet Data (e.g., Recent Political and Covid Pandemic Tweets)</td>
    <td align="center">172</td>
    <td align="center">59</td>
    <td align="center">0.826</td>
    <td align="center">0.869</td>
    <td align="center">0.826</td>
    <td align="center">0.836</td>
  </tr>
</table>

### Parameter Efficient Fine Tuning (PEFT) QLoRA, Model: Microsoft BERT (Encoder)

### Evaluation Dataset Scores

| Metric     | Value        |
|------------|--------------|
| Accuracy   | 0.9176       |
| F1 Score   | 0.8399       |
| Precision  | 0.7611       |
| Recall     | 0.9370       |


## Arabic Data

### 1. Voting Classifier (4 Models: 'dt', 'knn', 'rf', 'xgb')



<table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not Checkworthy</th>
    <th>Checkworthy</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Benchmark Data</td>
    <td align="center">17088</td>
    <td align="center">5413</td>
    <td align="center">0.934</td>
    <td align="center">0.934</td>
    <td align="center">0.934</td>
    <td align="center">0.931</td>
  </tr>
  <tr>
    <td>Resampled (Under-sampled) Data</td>
    <td align="center">9896</td>
    <td align="center">1254</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td>Tweet Data (e.g., Recent Political and Covid Pandemic Tweets)</td>
    <td align="center">172</td>
    <td align="center">59</td>
    <td align="center">0.895</td>
    <td align="center">0.898</td>
    <td align="center">0.895</td>
    <td align="center">0.896</td>
  </tr>
</table>
  



### 2. Voting Classifier (6 Models: 'rf', 'xgb', 'dt', 'knn', 'lgbm', 'adb')

<!--| Metric     | Value |
|------------|-------|
| Accuracy   | 0.801 |
| Precision  | 0.793 |
| Recall     | 0.801 |
| F1 Score   | 0.796 |-->


<table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not Checkworthy</th>
    <th>Checkworthy</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Benchmark Data</td>
    <td align="center">17088</td>
    <td align="center">5413</td>
    <td align="center">0.946</td>
    <td align="center">0.947</td>
    <td align="center">0.946</td>
    <td align="center">0.945</td>
  </tr>
  <tr>
    <td>Resampled (Under-sampled) Data</td>
   <td align="center">9896</td>
    <td align="center">1254</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td>Tweet Data (e.g., Recent Political and Covid Pandemic Tweets)</td>
    <td align="center">172</td>
    <td align="center">59</td>
    <td align="center">0.924</td>
    <td align="center">0.924</td>
    <td align="center">0.924</td>
    <td align="center">0.924</td>
  </tr>
</table>

### 3. Voting Classifier (3 Models: 'gdb', 'lgbm', 'adb')

<!--| Metric     | Value |
|------------|-------|
| Accuracy   | 0.801 |
| Precision  | 0.793 |
| Recall     | 0.801 |
| F1 Score   | 0.796 |-->


<table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not Checkworthy</th>
    <th>Checkworthy</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Benchmark Data</td>
    <td align="center">17088</td>
    <td align="center">5413</td>
    <td align="center">0.943</td>
    <td align="center">0.945</td>
    <td align="center">0.943</td>
    <td align="center">0.941</td>
  </tr>
  <tr>
    <td>Resampled (Under-sampled) Data</td>
   <td align="center">9896</td>
    <td align="center">1254</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td>Tweet Data (e.g., Recent Political and Covid Pandemic Tweets)</td>
    <td align="center">172</td>
    <td align="center">59</td>
    <td align="center">0.925</td>
    <td align="center">0.925</td>
    <td align="center">0.925</td>
    <td align="center">0.925</td>
  </tr>
</table>

### 4. Voting Classifier (8 Hyperparameter Tuned Models: 'dt', 'knn', 'rf', 'xgb', 'adb')

<!--| Metric     | Value |
|------------|-------|
| Accuracy   | 0.801 |
| Precision  | 0.793 |
| Recall     | 0.801 |
| F1 Score   | 0.796 |-->


<table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not Checkworthy</th>
    <th>Checkworthy</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Benchmark Data</td>
    <td align="center">17088</td>
    <td align="center">5413</td>
    <td align="center">0.934</td>
    <td align="center">0.936</td>
    <td align="center">0.934</td>
    <td align="center">0.931</td>
  </tr>
  <tr>
    <td>Resampled (Under-sampled) Data</td>
    <td align="center">9896</td>
    <td align="center">1254</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td>Tweet Data (e.g., Recent Political and Covid Pandemic Tweets)</td>
    <td align="center">172</td>
    <td align="center">59</td>
    <td align="center">0.895</td>
    <td align="center">0.898</td>
    <td align="center">0.895</td>
    <td align="center">0.896</td>
  </tr>
</table>

### 5. Voting Classifier (3 Models: 'rf', 'xgb', 'dt')

<!--| Metric     | Value |
|------------|-------|
| Accuracy   | 0.801 |
| Precision  | 0.793 |
| Recall     | 0.801 |
| F1 Score   | 0.796 |-->


<table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not Checkworthy</th>
    <th>Checkworthy</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Benchmark Data</td>
    <td align="center">17088</td>
    <td align="center">5413</td>
    <td align="center">0.953</td>
    <td align="center">0.953</td>
    <td align="center">0.953</td>
    <td align="center">0.953</td>
  </tr>
  <tr>
    <td>Resampled (Under-sampled) Data</td>
   <td align="center">9896</td>
    <td align="center">1254</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td>Tweet Data (e.g., Recent Political and Covid Pandemic Tweets)</td>
    <td align="center">172</td>
    <td align="center">59</td>
    <td align="center">0.914</td>
    <td align="center">0.919</td>
    <td align="center">0.914</td>
    <td align="center">0.916</td>
  </tr>
</table>

### 6. Voting Classifier (4 Models: 'xgb', 'lgbm', 'adb', 'lr')

<!--| Metric     | Value |
|------------|-------|
| Accuracy   | 0.801 |
| Precision  | 0.793 |
| Recall     | 0.801 |
| F1 Score   | 0.796 |-->


<table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not Checkworthy</th>
    <th>Checkworthy</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Benchmark Data</td>
    <td align="center">17088</td>
    <td align="center">5413</td>
    <td align="center">0.958</td>
    <td align="center">0.958</td>
    <td align="center">0.958</td>
    <td align="center">0.958</td>
  </tr>
  <tr>
    <td>Resampled (Under-sampled) Data</td>
   <td align="center">9896</td>
    <td align="center">1254</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td>Tweet Data (e.g., Recent Political and Covid Pandemic Tweets)</td>
    <td align="center">172</td>
    <td align="center">59</td>
    <td align="center">0.919</td>
    <td align="center">0.924</td>
    <td align="center">0.919</td>
    <td align="center">0.921</td>
  </tr>
</table>

### 7. Voting Classifier (8 Hyperparameter Tuned Models: 'dt', 'knn', 'rf', 'xgb', 'gdb', 'lgbm', 'adb', 'lr')

<!--| Metric     | Value |
|------------|-------|
| Accuracy   | 0.801 |
| Precision  | 0.793 |
| Recall     | 0.801 |
| F1 Score   | 0.796 |-->


<table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not Checkworthy</th>
    <th>Checkworthy</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Benchmark Data</td>
    <td align="center">17088</td>
    <td align="center">5413</td>
    <td align="center">0.957</td>
    <td align="center">0.957</td>
    <td align="center">0.957</td>
    <td align="center">0.956</td>
  </tr>
  <tr>
    <td>Resampled (Under-sampled) Data</td>
   <td align="center">9896</td>
    <td align="center">1254</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td>Tweet Data (e.g., Recent Political and Covid Pandemic Tweets)</td>
    <td align="center">172</td>
    <td align="center">59</td>
    <td align="center">0.926</td>
    <td align="center">0.928</td>
    <td align="center">0.926</td>
    <td align="center">0.927</td>
  </tr>
</table>

### Blending Classifier ('rf', 'xgb', 'dt', 'knn', 'lgbm', 'adb')

<table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not Checkworthy</th>
    <th>Checkworthy</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Benchmark Data</td>
    <td align="center">17088</td>
    <td align="center">5413</td>
    <td align="center">0.956</td>
    <td align="center">0.956</td>
    <td align="center">0.956</td>
    <td align="center">0.955</td>
  </tr>
  <tr>
    <td>Resampled (Under-sampled) Data</td>
    <td align="center">9896</td>
    <td align="center">1254</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td>Tweet Data (e.g., Recent Political and Covid Pandemic Tweets)</td>
    <td align="center">172</td>
    <td align="center">59</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
</table>



### Stacking Classifier (Working)

### Random Forest Classifier (RFC)

<table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not Checkworthy</th>
    <th>Checkworthy</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Benchmark Data</td>
    <td align="center">17088</td>
    <td align="center">5413</td>
    <td align="center">0.825</td>
    <td align="center">0.858</td>
    <td align="center">0.825</td>
    <td align="center">0.834</td>
  </tr>
  <tr>
    <td>Resampled (Under-sampled) Data</td>
    <td align="center">9896</td>
    <td align="center">1254</td>
    <td align="center">0.745</td>
    <td align="center">0.728</td>
    <td align="center">0.745</td>
    <td align="center">0.734</td>
  </tr>
  <tr>
    <td>Tweet Data (e.g., Recent Political and Covid Pandemic Tweets)</td>
    <td align="center">172</td>
    <td align="center">59</td>
    <td align="center">0.841</td>
    <td align="center">0.838</td>
    <td align="center">0.841</td>
    <td align="center">0.840</td>
  </tr>
</table>




### Decision Tree Classifier (DT)

<!--| Metric     | Value |
|------------|-------|
| Accuracy   | 0.721 |
| Precision  | 0.729 |
| Recall     | 0.721 |
| F1 Score   | 0.725 |-->

<table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not Checkworthy</th>
    <th>Checkworthy</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Benchmark Data</td>
    <td align="center">5090</td>
    <td align="center">2243</td>
    <td align="center">0.635</td>
    <td align="center">0.788</td>
    <td align="center">0.635</td>
    <td align="center">0.663</td>
  </tr>
 <!-- <tr>
    <td>Resampled (Under-sampled) Data</td>
    <td align="center">9896</td>
    <td align="center">1254</td>
    <td align="center">0.721</td>
    <td align="center">0.729</td>
    <td align="center">0.721</td>
    <td align="center">0.725</td>
  </tr>-->
 <!-- <tr>
    <td>Tweet Data (e.g., Recent Political and Covid Pandemic Tweets)</td>
    <td align="center">172</td>
    <td align="center">59</td>
    <td align="center">0.763</td>
    <td align="center">0.822</td>
    <td align="center">0.763</td>
    <td align="center">0.779</td>
  </tr>-->
</table>



### K-Nearest Neighbors (KNN)

<!--| Metric     | Value |
|------------|-------|
| Accuracy   | 0.674 |
| Precision  | 0.666 |
| Recall     | 0.674 |
| F1 Score   | 0.670 |-->

<table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not Checkworthy</th>
    <th>Checkworthy</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Benchmark Data</td>
    <td align="center">17088</td>
    <td align="center">5413</td>
    <td align="center">0.768</td>
    <td align="center">0.732</td>
    <td align="center">0.768</td>
    <td align="center">0.736</td>
  </tr>
  <tr>
    <td>Resampled (Under-sampled) Data</td>
    <td align="center">9896</td>
    <td align="center">1254</td>
    <td align="center">0.674</td>
    <td align="center">0.666</td>
    <td align="center">0.674</td>
    <td align="center">0.670</td>
  </tr>
  <tr>
    <td>Tweet Data (e.g., Recent Political and Covid Pandemic Tweets)</td>
    <td align="center">172</td>
    <td align="center">59</td>
    <td align="center">0.673</td>
    <td align="center">0.691</td>
    <td align="center">0.673</td>
    <td align="center">0.681</td>
  </tr>
</table>




### XGBoost Classifier (XGB)

<!--| Metric     | Value |
|------------|-------|
| Accuracy   | 0.795 |
| Precision  | 0.805 |
| Recall     | 0.795 |
| F1 Score   | 0.799 |-->

<table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not Checkworthy</th>
    <th>Checkworthy</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Benchmark Data</td>
    <td align="center">17088</td>
    <td align="center">5413</td>
    <td align="center">0.957</td>
    <td align="center">0.957</td>
    <td align="center">0.957</td>
    <td align="center">0.956</td>
  </tr>
  <tr>
    <td>Resampled (Under-sampled) Data</td>
    <td align="center">9896</td>
    <td align="center">1254</td>
    <td align="center">0.795</td>
    <td align="center">0.805</td>
    <td align="center">0.795</td>
    <td align="center">0.799</td>
  </tr>
  <tr>
    <td>Tweet Data (e.g., Recent Political and Covid Pandemic Tweets)</td>
    <td align="center">172</td>
    <td align="center">59</td>
    <td align="center">0.925</td>
    <td align="center">0.930</td>
    <td align="center">0.925</td>
    <td align="center">0.926</td>
  </tr>
</table>

### Gradient Boosting Classifier (GDB)

<!--| Metric     | Value |
|------------|-------|
| Accuracy   | 0.795 |
| Precision  | 0.805 |
| Recall     | 0.795 |
| F1 Score   | 0.799 |-->

<table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not Checkworthy</th>
    <th>Checkworthy</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Benchmark Data</td>
    <td align="center">17088</td>
    <td align="center">5413</td>
    <td align="center">0.957</td>
    <td align="center">0.957</td>
    <td align="center">0.957</td>
    <td align="center">0.956</td>
  </tr>
  <tr>
    <td>Resampled (Under-sampled) Data</td>
    <td align="center">9896</td>
    <td align="center">1254</td>
    <td align="center">0.795</td>
    <td align="center">0.805</td>
    <td align="center">0.795</td>
    <td align="center">0.799</td>
  </tr>
  <tr>
    <td>Tweet Data (e.g., Recent Political and Covid Pandemic Tweets)</td>
    <td align="center">172</td>
    <td align="center">59</td>
    <td align="center">0.910</td>
    <td align="center">0.911</td>
    <td align="center">0.910</td>
    <td align="center">0.911</td>
  </tr>
</table>

### Light GBM Classifier

<!--| Metric     | Value |
|------------|-------|
| Accuracy   | 0.795 |
| Precision  | 0.805 |
| Recall     | 0.795 |
| F1 Score   | 0.799 |-->

<table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not Checkworthy</th>
    <th>Checkworthy</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Benchmark Data</td>
    <td align="center">17088</td>
    <td align="center">5413</td>
    <td align="center">0.957</td>
    <td align="center">0.957</td>
    <td align="center">0.957</td>
    <td align="center">0.956</td>
  </tr>
  <tr>
    <td>Resampled (Under-sampled) Data</td>
    <td align="center">9896</td>
    <td align="center">1254</td>
    <td align="center">0.795</td>
    <td align="center">0.805</td>
    <td align="center">0.795</td>
    <td align="center">0.799</td>
  </tr>
  <tr>
    <td>Tweet Data (e.g., Recent Political and Covid Pandemic Tweets)</td>
    <td align="center">172</td>
    <td align="center">59</td>
    <td align="center">0.921</td>
    <td align="center">0.921</td>
    <td align="center">0.921</td>
    <td align="center">0.921</td>
  </tr>
</table>

### Ada Boost Classifier

<!--| Metric     | Value |
|------------|-------|
| Accuracy   | 0.795 |
| Precision  | 0.805 |
| Recall     | 0.795 |
| F1 Score   | 0.799 |-->

<table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not Checkworthy</th>
    <th>Checkworthy</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Benchmark Data</td>
    <td align="center">17088</td>
    <td align="center">5413</td>
    <td align="center">0.957</td>
    <td align="center">0.957</td>
    <td align="center">0.957</td>
    <td align="center">0.956</td>
  </tr>
  <tr>
    <td>Resampled (Under-sampled) Data</td>
    <td align="center">9896</td>
    <td align="center">1254</td>
    <td align="center">0.795</td>
    <td align="center">0.805</td>
    <td align="center">0.795</td>
    <td align="center">0.799</td>
  </tr>
  <tr>
    <td>Tweet Data (e.g., Recent Political and Covid Pandemic Tweets)</td>
    <td align="center">172</td>
    <td align="center">59</td>
    <td align="center">0.906</td>
    <td align="center">0.904</td>
    <td align="center">0.906</td>
    <td align="center">0.904</td>
  </tr>
</table>

### Logistic Regression

<!--| Metric     | Value |
|------------|-------|
| Accuracy   | 0.795 |
| Precision  | 0.805 |
| Recall     | 0.795 |
| F1 Score   | 0.799 |-->

<table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not Checkworthy</th>
    <th>Checkworthy</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Benchmark Data</td>
    <td align="center">17088</td>
    <td align="center">5413</td>
    <td align="center">0.957</td>
    <td align="center">0.957</td>
    <td align="center">0.957</td>
    <td align="center">0.956</td>
  </tr>
  <tr>
    <td>Resampled (Under-sampled) Data</td>
    <td align="center">9896</td>
    <td align="center">1254</td>
    <td align="center">0.795</td>
    <td align="center">0.805</td>
    <td align="center">0.795</td>
    <td align="center">0.799</td>
  </tr>
  <tr>
    <td>Tweet Data (e.g., Recent Political and Covid Pandemic Tweets)</td>
    <td align="center">172</td>
    <td align="center">59</td>
    <td align="center">0.826</td>
    <td align="center">0.869</td>
    <td align="center">0.826</td>
    <td align="center">0.836</td>
  </tr>
</table>

### Parameter Efficient Fine Tuning (PEFT) QLoRA, Model: Microsoft BERT (Encoder)

### Evaluation Dataset Scores

| Metric     | Value        |
|------------|--------------|
| Accuracy   | 0.9176       |
| F1 Score   | 0.8399       |
| Precision  | 0.7611       |
| Recall     | 0.9370       |

## This result has already surpassed the best scores from CLEF 2024, as seen [here](https://checkthat.gitlab.io/clef2024/task1/).

[checkThat](https://checkthat.gitlab.io/clef2024/task1/)
<img width="506" alt="image" src="https://github.com/user-attachments/assets/e5e0c9f1-8f0a-485d-9d55-04adc89a4833" />

