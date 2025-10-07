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
- gemini-1.5-flash - prompt engineering for class label automation for scrapped tweet .
<!-- falcon-7b - data pruning by using prompt engineering.-->
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

# English
## [Data Used](https://gitlab.com/checkthat_lab/clef2024-checkthat-lab/-/tree/main/task1/data)
<Table border="1">
  <tr>
    <th>Dataset</th>
    <th>Not CheckWorthy</th>
    <th>CheckWorthy</th>
  </tr>
  <td>Training Data</td>
  <td align="center">17087</td>
  <td align="center">5413</td>
</tr> 
  <tr>
  <td>UnderSampled Training Data</td>
  <td align="center">7189</td>
  <td align="center"> 5408</td>
</tr>
<tr>
  <td>Dev Data (hyperparametr Tuning)</td>
  <td align="center">794</td>
  <td align="center"> 238</td>
</tr>

<tr>
  <td>Dev Test Data (Model Test)</td>
  <td align="center">210</td>
  <td align="center"> 108</td>
</tr>
<tr>
  <td>Test Data (Model Test)</td>
  <td align="center">253</td>
  <td align="center"> 88</td>
</tr>

<tr>
  <td>Tweet Data (Model Test)</td>
  <td align="center">124</td>
  <td align="center">107</td>
</tr>
</Table>

## Trained on Benchmark DataSet and tested on Test data
<table border="1">
  <tr>
    <th>Models</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th
  </tr>
  <tr>
    <th>XGBoost</th>
    <th>0.830</th>
    <th>0.8222</th>
    <th>0.830</th>
    <th>0.821</th>
  </tr>
  <tr>
    <th>Gradient Boosting</th>
    <th>0.798</th>
    <th>0.789</th>
    <th>0.798</th>
    <th>0.768</th>
  </tr>

  
   <tr>
    <th>Light GBM</th>
    <th>0.8333</th>
    <th>0.837</th>
    <th>0.833</th>
    <th>0.8111</th>
  </tr>

  <tr>
    <th>Logistic Regression</th>
    <th>0.748</th>
    <th>0.816</th>
    <th>0.748</th>
    <th>0.763</th>
</th>
  </tr>

  <tr>
    <th>Voating Classifier (Soft: XGB + GDB + LR)</th>
    <th>0.833</th>
    <th>0.825</th>
    <th>0.833</th>
    <th>0.825</th>
  </tr>

   <tr>
    <th>Voating Classifier (Hard: XGB + LGBM + ADB + LR)</th>
    <th>0.836</th>
    <th>0.850</th>
    <th>0.836</th>
    <th>0.811</th>
  </tr>
  
 <tr>
    <th>Voating Classifier (Hard: XGB + GDB + LR)</th>
    <th>0.878</th>
    <th>0.881</th>
    <th>0.877</th>
    <th>0.878</th>
  </tr>

  
</table>

## Performance of classicial ML models on Undersampled English Dataset and tested on dev test dataset

<table border = "1">
 <tr>
   <th>Model</th>
   <th>Accuracy</th>
   <th>Precision</th>
   <th>Recall</th>
   <th>F1</th>
 </tr> 
  
  <tr>
   <th>Decision Tree</th>
   <th>0.709</th>
   <th>0.714</th>
   <th>0.709</th>
   <th>0.711</th>
  </tr> 
  
  <tr>
      <th>KNN</th>
   <th>0.671</th>
   <th>0.657</th>
   <th>0.671</th>
   <th>0.660</th>
  </tr> 
  
  <tr>
      <th>XGBoost</th>
   <th>0.867</th>
   <th>0.872</th>
   <th>0.867</th>
   <th>0.868</th>
  </tr> 
  
  <tr>
      <th>Random Forest</th>
   <th>0.728</th>
   <th>0.717</th>
   <th>0.728</th>
   <th>0.714</th>
  </tr> 
  
  <tr>
      <th>Gradient Boosting</th>
   <th>0.861</th>
   <th>0.860</th>
   <th>0.861</th>
   <th>0.857</th>
  </tr> 
  
  <tr>
      <th>Light GBM</th>
   <th>0.845</th>
   <th>0.843</th>
   <th>0.845</th>
   <th>0.842</th>
  </tr> 
  
  <tr>
      <th>Ada Boost</th>
   <th>0.813</th>
   <th>0.815</th>
   <th>0.813</th>
   <th>0.803</th>
  </tr> 
  
  <tr>
      <th>Logistic Regression</th>
   <th>0.851</th>
   <th>0.874</th>
   <th>0.851</th>
   <th>0.855</th>
  </tr> 
  
  <tr>
      <th>Voating Classifier (Soft: XGb + GDB + LR)</th>
   <th>0.884</th>
   <th>0.889</th>
   <th>0.883</th>
   <th>0.884</th>
  </tr> 
  
  <tr>
      <th>Voating Classifier (Hard: XGb + LGBM + ADb  + LR)</th>
   <th>0.864</th>
   <th>0.863</th>
   <th>0.864</th>
   <th>0.862</th>
  </tr> 
  
  <tr>
      <th>Voating Classifier (Hard: XGb + GDB + LR)</th>
   <th>0.877</th>
   <th>0.881</th>
   <th>0.877</th>
   <th>0.878</th>
  </tr> 
</table>

## Performance of different models on the original dataset and evaluated on dev test data

<table border = "1">
 <tr>
   <th>Model</th>
   <th>Accuracy</th>
   <th>Precision</th>
   <th>Recall</th>
   <th>F1</th>
 </tr> 
  
  <tr>
   <th>Decision Tree</th>
   <th>0.718</th>
   <th>0.720</th>
   <th>0.718</th>
   <th>0.719</th>
  </tr> 
  
  <tr>
      <th>KNN</th>
   <th>0.687</th>
   <th>0.685</th>
   <th>0.687</th>
   <th>0.615</th>
  </tr> 
  
  <tr>
      <th>XGBoost</th>
   <th>0.791</th>
   <th>0.824</th>
   <th>0.791</th>
   <th>0.764</th>
  </tr> 
  
  <tr>
      <th>Random Forest</th>
   <th>0.753</th>
   <th>0.746</th>
   <th>0.753</th>
   <th>0.746</th>
  </tr> 
  
  <tr>
      <th>Gradient Boosting</th>
   <th>0.801</th>
   <th>0.831</th>
   <th>0.801</th>
   <th>0.777</th>
  </tr> 
  
  <tr>
      <th>Light GBM</th>
   <th>0.807</th>
   <th>0.828</th>
   <th>0.807</th>
   <th>0.788</th>
  </tr> 
  
  <tr>
      <th>Ada Boost</th>
   <th>0.785</th>
   <th>0.814</th>
   <th>0.785</th>
   <th>0.757</th>
  </tr> 
  
  <tr>
      <th>Logistic Regression</th>
   <th>0.851</th>
   <th>0.857</th>
   <th>0.851</th>
   <th>0.844</th>
  </tr> 
  
  <tr>
      <th>Voating Classifier (Soft: XGb + GDB + LR)</th>
   <th>0.813</th>
   <th>0.840</th>
   <th>0.813</th>
   <th>0.793</th>
  </tr> 
</table>

## Performance of different classifiers on the undersampled dataset (evaluated on test data):

<table border = "1">
 <tr>
   <th>Model</th>
   <th>Accuracy</th>
   <th>Precision</th>
   <th>Recall</th>
   <th>F1</th>
 </tr> 
  
  <tr>
   <th>KNN</th>
   <th>0.669</th>
   <th>0.641</th>
   <th>0.669</th>
   <th>0.653</th>
  </tr> 
  
  <tr>
      <th>XGBoost</th>
   <th>0.718</th>
   <th>0.785</th>
   <th>0.718</th>
   <th>0.735</th>
  </tr> 
  
  <tr>
      <th>Gradient Boosting</th>
   <th>0.780</th>
   <th>0.790</th>
   <th>0.780</th>
   <th>0.784</th>
  </tr> 
  
  <tr>
      <th>Light GBM</th>
   <th>0.812</th>
   <th>0.812</th>
   <th>0.812</th>
   <th>0.812</th>
  </tr> 
  
  <tr>
      <th>Ada Boost</th>
   <th>0.7333</th>
   <th>0.701</th>
   <th>0.733</th>
   <th>0.709</th>
  </tr> 
  
  <tr>
      <th>Logistic Regression</th>
   <th>0.724</th>
   <th>0.814</th>
   <th>0.724</th>
   <th>0.741</th>
  </tr> 
  
  <tr>
      <th>Voating Classifier (Soft: XGB + GDB + LR)</th>
   <th>0.777</th>
   <th>0.818</th>
   <th>0.777</th>
   <th>0.788</th>
  </tr> 
  
  
  
  <tr>
      <th>Voating Classifier (Hard: XGb +  LGBM + ADB + LR)</th>
   <th>0.806</th>
   <th>0.808</th>
   <th>0.806</th>
   <th>0.807</th>
  </tr> 
</table>

## Performance of classifier trained on undersampled dataset , evaluated on tweet data

### Query for tweet data scrapping:
``QUERY = "COVID OR vaccine OR pandemic OR lockdown OR #COVID19 OR " \
        "#VaccineMandate elections OR Biden OR Trump OR Joe Biden OR " \
        "DOGE OR FBI OR Donald Trump OR Ukraine OR Russia OR " \
        "Middle East Crisis OR South Asia Crisis OR UN meeting OR" \
        " US Congress OR US Republic OR geopolitics OR War OR" \
        " #Politics OR #Election2025  OR #UkraineRussiaWar OR" \
        " #Trump OR #DOGE OR #Trump2025 #MiddleEastCrisis OR" \
        " #Geopolitics OR #USRepublic OR #USCongress OR #Bangladesh (COVID OR" \
        " OR OR vaccine OR OR OR pandemic OR OR OR lockdown OR OR OR" \
        " #COVID19 OR OR OR #VaccineMandate OR elections OR OR OR" \
        " Biden OR OR OR Trump OR OR OR Joe OR Biden OR OR OR" \
        " DOGE OR OR OR FBI OR OR OR Donald OR Trump OR OR OR" \
        " Ukraine OR OR OR Russia OR OR OR Middle OR " \
        "East OR Crisis OR OR OR South OR Asia OR Crisis OR OR OR" \
        " UN OR meeting OR OR OR US OR Congress OR OR OR US OR" \
        " Republic OR OR OR geopolitics OR OR OR War OR OR OR" \
        " #Politics OR OR OR #Election2025 OR OR OR #UkraineRussiaWar OR OR OR" \
        " #Trump OR OR OR #DOGE OR OR OR #Trump2025 OR #MiddleEastCrisis OR OR OR" \
        " #Geopolitics OR OR OR #USRepublic OR OR OR #USCongress OR OR OR #Bangladesh)" \
        " lang:en until:2022-12-31 since:2020-01-01"
        ``
<table border = "1">
 <tr>
   <th>Model</th>
   <th>Accuracy</th>
   <th>Precision</th>
   <th>Recall</th>
   <th>F1</th>
 </tr> 
  
  <tr>
   <th>Decision Tree</th>
   <th>0.641</th>
   <th>0.698</th>
   <th>0.641</th>
   <th>0.625</th>
  </tr> 
  
  <tr>
      <th>XGBoost</th>
   <th>0.623</th>
   <th>0.659</th>
   <th>0.623</th>
   <th>0.613</th>
  </tr> 
  
  <tr>
      <th>Gradient Boosting</th>
   <th>0.654</th>
   <th>0.653</th>
   <th>0.654</th>
   <th>0.653</th>
  </tr> 
  
  <tr>
      <th>Light GBM</th>
   <th>0.649</th>
   <th>0.648</th>
   <th>0.649</th>
   <th>0.648</th>
  </tr> 
  
  <tr>
      <th>Ada Boost</th>
   <th>0.632</th>
   <th>0.639</th>
   <th>0.632</th>
   <th>0.616</th>
  </tr> 
  
  <tr>
      <th>Logistic Regression</th>
   <th>0.688</th>
   <th>0.740</th>
   <th>0.688</th>
   <th>0.678</th>
  </tr> 
  
  <tr>
      <th>Voating Classifier (Soft: XGB + GDB + LR)</th>
   <th>0.658</th>
   <th>0.675</th>
   <th>0.658</th>
   <th>0.656</th>
  </tr> 
  
  
  
  <tr>
      <th>Voating Classifier (Hard: XGb +  LGBM + ADB + LR)</th>
   <th>0.671</th>
   <th>0.670</th>
   <th>0.671</th>
   <th>0.670</th>
  </tr> 
</table>
        


## Performance of LLMs for english dataset across different evaluation datasets:
<table border = "1">
 <tr>
   <th>Model</th>
   <th>Dataset</th>
   <th>Accuracy</th>
   <th>Precision</th>
   <th>F1</th>
 </tr> 
  
  <tr>
   <th>DBERTa (epoch : 8)</th>
   <th>Test Data</th>
   <th>0.836</th>
   <th>0.834</th>
   <th>0.819</th>
  </tr> 
  
  <tr>
   <th>DBERTa (epoch : 8)</th>
   <th>Dev Test Data</th>
   <th>0.851</th>
   <th>0.861</th>
   <th>0.843</th>
  </tr>

  <tr>
   <th>DBERTa (epoch : 8)</th>
   <th>Tweet Data</th>
   <th>0.779</th>
   <th>0.788</th>
   <th>0.775</th>
  </tr>
  
  <tr>
      <th>BERT (QLoRA, Undersampled, Hyperparametered tuned with optuna, 40 epoch)</th>
   <th>Test Data</th>
   <th>0.839</th>
   <th>0.834</th>
   <th>0.825</th>
  </tr> 

  <tr>
      <th>BERT (QLoRA, Undersampled, Hyperparametered tuned with optuna, 40 epoch)</th>
   <th>Dev Test Data</th>
   <th>0.804</th>
   <th>0.819</th>
   <th>0.8786</th>
  </tr> 

  <tr>
      <th>BERT (QLoRA, Undersampled, Hyperparametered tuned with optuna, 40 epoch)</th>
   <th>Tweet Data</th>
   <th>0.745</th>
   <th>0.750</th>
   <th>0.741</th>
  </tr> 
  
  <tr>
      <th>RoBERTa - Base (25 epochs)</th>
   <th>Test Data</th>
   <th>0.821</th>
   <th>0.818</th>
   <th>0.819</th>
  </tr> 

   <tr>
      <th>RoBERTa - Base (25 epochs)</th>
   <th>Dev Test Data</th>
   <th>0.848</th>
   <th>0.847</th>
   <th>0.847</th>
  </tr> 

   <tr>
      <th>RoBERTa - Base (25 epochs)</th>
   <th>Tweet Data</th>
   <th>0.736</th>
   <th>0.739</th>
   <th>0.736</th>
  </tr> 
  
  
</table>

<!-- ###########################################################################################################################--->
# Dutch

## [Data Used](https://gitlab.com/checkthat_lab/clef2024-checkthat-lab/-/tree/main/task1/data)
<table border="1">
<tr>
  <th>Dataset</th>
  <th>CheckWorthy</th>
  <th> Not Checkworthy</th>
</tr>
<tr>
  <td>Training Data</td>
  <td align="center">405</td>
  <td align="center">590</td>
</tr> 
<tr>
  <td>Dev Data (hyperparametr Tuning)</td>
  <td align="center">102</td>
  <td align="center"> 150 </td>
</tr>

<tr>
  <td>Dev Test Data (Model Test)</td>
  <td align="center">316</td>
  <td align="center"> 350 </td>
</tr>

<tr>
  <td>Test Data (Model Test)</td>
  <td align="center">397</td>
  <td align="center">603</td>
</tr>
</table>

## Result
<table border="1">
  <tr>
    <th>Models</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <th>KNN Classifier</th>
    <th>0.518</th>
    <th>0.0.521</th>
    <th>0.0.518</th>
    <th>0.0.518</th>
  </tr>
  <tr>
    <th>Decision Tree Classifier</th>
    <th>0.538</th>
    <th>0.545</th>
    <th>0.538</th>
    <th>0.534</th>
  </tr>

  
   <tr>
    <th>Light GBM</th>
    <th>0.506</th>
    <th>0.506</th>
    <th>0.506</th>
    <th>0.506</th>
  </tr>

  <tr>
    <th>Gradient Boosting</th>
    <th>0.498</th>
    <th>0.493</th>
    <th>0.498</th>
    <th>0.492</th>
  </tr>

  <tr>
    <th>Random Forest</th>
    <th>0.494</th>
    <th>0.500</th>
    <th>0.494</th>
    <th>0.489</th>
  </tr>

   <tr>
    <th>Voting(Soft: XGB + GDB + LR)</th>
    <th>0.505</th>
    <th>0.507</th>
    <th>0.505</th>
    <th>0.504</th>
  </tr>

  <tr>
    <th>Voating(Soft: decision tree + KNN + random forest + XGB)</th>
    <th>0.508</th>
    <th>0.522</th>
    <th>0.508</th>
    <th>0.490</th>
  </tr>

  <tr>
    <th>Voating (Hard: Decision TRee + KNN  + Random Forest + XGB)</th>
    <th>0.520</th>
    <th>0.525</th>
    <th>0.520</th>
    <th>0.517</th>
  </tr>

</table>






<!-- ############################################################################################################################--->

# Arabic

## [Data Used](https://gitlab.com/checkthat_lab/clef2024-checkthat-lab/-/tree/main/task1/data)
<table border="1">
<tr>
  <th>Dataset</th>
  <th>Not CheckWorthy</th>
  <th>Checkworthy</th>
</tr>
<tr>
  <td>Training Data</td>
  <td align="center">5090</td>
  <td align="center">2243</td>
</tr> 
<tr>
  <td>Dev Data (hyperparametr Tuning)</td>
  <td align="center">682</td>
  <td align="center"> 411 </td>
</tr>

<tr>
  <td>Dev Test Data (Model Test)</td>
  <td align="center">377</td>
  <td align="center"> 123 </td>
</tr>
</table>

## Result
<table border="1">
  <tr>
    <th>Models</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
    <th>Confusion Matrix</th>
  </tr>
  <tr>
    <th>Decision Tree Classifier</th>
    <th>0.911</th>
    <th>0.912</th>
    <th>0.911</th>
    <th>0.911</th>
    <th><img width="1080" height="960" alt="Image" src="https://github.com/user-attachments/assets/c8de8650-6ba9-4f79-8271-80ec120ddcfe" /></th>
  </tr>
  <tr>
    <th>KNN Classifier</th>
    <th>1.00</th>
    <th>1.00</th>
    <th>1.00</th>
    <th>1.00</th>
    <th><img width="1080" height="960" alt="Image" src="https://github.com/user-attachments/assets/c8506364-b6ae-4faa-9e15-e05db83e5af9" /></th>
  </tr>

  
   <tr>
    <th>XGB Classifier</th>
    <th>0.936</th>
    <th>0.945</th>
    <th>0.936</th>
    <th>0.937</th>
     <th><img width="1080" height="960" alt="Image" src="https://github.com/user-attachments/assets/c8202113-44b9-45e0-9dc8-8f5b872f7587" /></th>
  </tr>

  <tr>
    <th>Random Forest Classifier</th>
    <th>0.924</th>
    <th>0.389</th>
    <th>0.624</th>
    <th>0.479</th>
    <th><img width="1080" height="960" alt="Image" src="https://github.com/user-attachments/assets/b995591c-323e-4814-9a6f-38b0e2c968cc" /></th>
  </tr>

  <tr>
    <th>Gradient Boosting Classifier</th>
    <th>0.995</th>
    <th>0.995</th>
    <th>0.995</th>
    <th>0.995</th>
    <th><img width="1080" height="960" alt="Image" src="https://github.com/user-attachments/assets/d5f2155d-b4b2-4213-bbf2-8d92611873e5" /></th>
  </tr>

   <tr>
    <th>Light GBM Classifier</th>
    <th>0.993</th>
    <th>0.993</th>
    <th>0.993</th>
    <th>0.993</th>
    <th><img width="1080" height="960" alt="Image" src="https://github.com/user-attachments/assets/f84bed07-b914-4796-9179-42a3733333d8" /></th>
  </tr>

  <tr>
    <th>Ada Boost Classifier</th>
    <th>0.743</th>
    <th>0.775</th>
    <th>0.743</th>
    <th>0.710</th>
    <th><img width="1080" height="960" alt="Image" src="https://github.com/user-attachments/assets/e60d7817-1c04-4466-b1c7-48a7683f9932" /></th>
  </tr>

  <tr>
    <th>Logistic Regression</th>
    <th>0.376</th>
    <th>0.141</th>
    <th>0.376</th>
    <th>0.206</th>
    <th><img width="1080" height="960" alt="Image" src="https://github.com/user-attachments/assets/bddf9b3d-b294-4ca7-9620-6e7488f9a8e9" /></th>
  </tr>

  <tr>
    <th>Voating Classifier, Ensembled Models: ('dt', 'knn', 'rf', 'xgb')</th>
    <th>0.979</th>
    <th>0.980</th>
    <th>0.979</th>
    <th>0.979</th>
    <th><img width="1080" height="960" alt="Image" src="https://github.com/user-attachments/assets/e00ff91a-519b-44a2-ba41-0e5144b9f315" /></th>
  </tr>

  <tr>
    <th>Voating Classifier, Ensembled Models: ('rf', 'xgb', 'dt', 'knn', 'lgbm', 'adb')</th>
    <th>0.997</th>
    <th>0.997</th>
    <th>0.997</th>
    <th>0.997</th>
    <th><img width="1080" height="960" alt="Image" src="https://github.com/user-attachments/assets/33bd3679-a030-4ac0-afc9-ff66008016ff" /></th>
  </tr>

  <tr>
    <th>Voating Classifier, Ensembled Models: ('gdb', 'lgbm', 'adb')</th>
    <th>0.998</th>
    <th>0.998</th>
    <th>0.998</th>
    <th>0.998</th>
    <th><img width="1080" height="960" alt="Image" src="https://github.com/user-attachments/assets/4d301094-48b6-450b-9a20-cbbd54dc3cd7" /></th>
  </tr>

  <tr>
    <th>Voating Classifier, Ensembled Models: ('dt', 'knn', 'rf', 'xgb', 'adb')</th>
    <th>0.981</th>
    <th>0.981</th>
    <th>0.981</th>
    <th>0.981</th>
    <th><img width="1080" height="960" alt="Image" src="https://github.com/user-attachments/assets/bd23d42d-abaa-4ad0-905f-7f966b3d27c8" /></th>
  </tr>


  <tr>
    <th>Voating Classifier, Ensembled Models: ('rf', 'xgb', 'dt')</th>
    <th>0.911</th>
    <th>0.912</th>
    <th>0.911</th>
    <th>0.911</th>
    <th><img width="1080" height="960" alt="image" src="https://github.com/user-attachments/assets/c7a01ac9-c1e1-4477-9b55-a620308a47ba" />
</th>
  </tr>

  <tr>
    <th>Voating Classifier, Ensembled Models: ('xgb', 'lgbm', 'adb', 'lr')</th>
    <th>0.997</th>
    <th>0.997</th>
    <th>0.997</th>
    <th>0.997</th>
    <th><img width="1080" height="960" alt="image" src="https://github.com/user-attachments/assets/1e14fe8b-24df-423a-a6f7-f336a81c9a7a" />
</th>
  </tr>

  <tr>
    <th>Voating Classifier, Ensembled Models: ('dt', 'knn', 'rf', 'xgb', 'gdb', 'lgbm', 'adb', 'lr', 'rf')</th>
    <th>0.999</th>
    <th>0.999</th>
    <th>0.999</th>
    <th>0.999</th>
    <th><img width="1080" height="960" alt="Image" src="https://github.com/user-attachments/assets/05bdad50-43dc-45ce-9fd6-5d0029f3ae69" /></th>
  </tr>
  
  <tr>
    <th>Blending Classifier, Base Models: ('rf', 'xgb', 'dt', 'knn', 'lgbm', 'adb'), Meta Model : ('lr')</th>
    <th>0.624</th>
    <th>0.389</th>
    <th>0.624</th>
    <th>0.479</th>
    <th><img width="1080" height="960" alt="Image" src="https://github.com/user-attachments/assets/b998c597-8e73-4a4b-822f-a12e4819f6b6" /></th>
  </tr>

  <tr>
    <th>Stacking Classifier, Base Models: ('rf', 'xgb', 'dt', 'knn', 'gdb', 'lgbm', 'adb'), Meta Model : ('lr')</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th></th>
  </tr>
</table>

<!-- ############################################################################################################################--->
# Spanish

## [Data Used](https://gitlab.com/checkthat_lab/clef2024-checkthat-lab/-/tree/main/task1/data)
<table border="1">
<tr>
  <th>Dataset</th>
  <th>Not CheckWorthy</th>
  <th>Checkworthy</th>
</tr>
<tr>
  <td>Training Data</td>
  <td align="center">16862</td>
  <td align="center">3182</td>
</tr> 
<tr>
  <td>Dev Data (hyperparametr Tuning)</td>
  <td align="center">4296</td>
  <td align="center">704</td>
</tr>

<tr>
  <td>Dev Test Data (Model Test)</td>
  <td align="center">4491</td>
  <td align="center">509</td>
</tr>
</table>

## Result
<table border="1">
  <tr>
    <th>Models</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
    <th>Confusion Matrix</th>
  </tr>
  <tr>
    <th>Decision Tree Classifier</th>
    <th>0.809</th>
    <th>0.801</th>
    <th>0.809</th>
    <th>0.805</th>
    <th><img width="1080" height="960" alt="image" src="https://github.com/user-attachments/assets/e11ac864-067c-4e3b-8d5d-5c76d39d9e9d" />
</th>
  </tr>
  <tr>
    <th>KNN Classifier</th>
    <th>0.789</th>
    <th>0.773</th>
    <th>0.789</th>
    <th>0.781</th>
    <th><img width="1080" height="960" alt="image" src="https://github.com/user-attachments/assets/c04b6964-2d49-4d19-a8ce-fe7ad334e3d8" />
</th>
  </tr>

  
   <tr>
    <th>XGB Classifier</th>
    <th>0.837</th>
    <th>0.808</th>
    <th>0.837</th>
    <th>0.819</th>
     <th><img width="1080" height="960" alt="image" src="https://github.com/user-attachments/assets/2809bed7-b55d-42b0-b9b3-7061e8adb5cb" />
</th>
  </tr>

  <tr>
    <th>Random Forest Classifier</th>
    <th>0.516</th>
    <th>0.829</th>
    <th>0.516</th>
    <th>0.582</th>
    <th><img width="1080" height="960" alt="image" src="https://github.com/user-attachments/assets/28e28983-3a10-4334-8c0e-cdf39924eb62" />
</th>
  </tr>

  <tr>
    <th>Gradient Boosting Classifier</th>
    <th>0.867</th>
    <th>0.839</th>
    <th>0.867</th>
    <th>0.834</th>
    <th><img width="1080" height="960" alt="image" src="https://github.com/user-attachments/assets/b24a946b-0aa3-4681-bd6a-808b775af509" />
</th>
  </tr>

   <tr>
    <th>Light GBM Classifier</th>
    <th>0.859</th>
    <th>0.816</th>
    <th>0.859</th>
    <th>0.817</th>
     <th><img width="1080" height="960" alt="image" src="https://github.com/user-attachments/assets/42f1b2a2-8dfd-4e87-9db6-a7d59e87bc8a" />
</th>
  </tr>

  <tr>
    <th>Ada Boost Classifier</th>
    <th>0.860</th>
    <th>0.816</th>
    <th>0.860</th>
    <th>0.805</th>
    <th><img width="1080" height="960" alt="image" src="https://github.com/user-attachments/assets/6fc348e4-8104-4190-89a6-6967c3ad378e" />
</th>
  </tr>

  <tr>
    <th>Logistic Regression</th>
    <th>0.791</th>
    <th>0.800</th>
    <th>0.791</th>
    <th>0.795</th>
    <th><img width="1080" height="960" alt="image" src="https://github.com/user-attachments/assets/8d0447bb-02e0-4957-bf86-b749c228d63a" />
</th>
  </tr>

  <tr>
    <th>Voating Classifier, Ensembled Models: ('dt', 'knn', 'rf', 'xgb')</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>

  <tr>
    <th>Voating Classifier, Ensembled Models: ('rf', 'xgb', 'dt', 'knn', 'lgbm', 'adb')</th>
   <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>

  <tr>
    <th>Voating Classifier, Ensembled Models: ('gdb', 'lgbm', 'adb')</th>
    <th>0.865</th>
    <th>0.838</th>
    <th>0.865</th>
    <th>0.821</th>
    <th><img width="1080" height="960" alt="image" src="https://github.com/user-attachments/assets/c95dbdcb-1586-45e2-bf5c-031b27fccaa4" />
</th>
  </tr>

  <tr>
    <th>Voating Classifier, Ensembled Models: ('dt', 'knn', 'rf', 'xgb', 'adb')</th>
    <th>0.847</th>
    <th>0.805</th>
    <th>0.847</th>
    <th>0.816</th>
    <th><img width="1080" height="960" alt="image" src="https://github.com/user-attachments/assets/2663cf0d-61c5-4ab9-85e7-5dd584cc9588" />
</th>
  </tr>


  <tr>
    <th>Voating Classifier, Ensembled Models: ('rf', 'xgb', 'dt')</th>
    <th>0.823</th>
    <th>0.804</th>
    <th>0.823</th>
    <th>0.812</th>
    <th><img width="1080" height="960" alt="image" src="https://github.com/user-attachments/assets/b4df726f-57d9-4408-89fb-803f87fec2af" />
</th>
  </tr>

  <tr>
    <th>Voating Classifier, Ensembled Models: ('xgb', 'lgbm', 'adb', 'lr')</th>
    <th>0.857</th>
    <th>0.820</th>
    <th>0.857</th>
    <th>0.825</th>
    <th><img width="1080" height="960" alt="image" src="https://github.com/user-attachments/assets/15606234-26cd-4614-a591-5495d9d278ec" />
</th>
  </tr>

  <tr>
    <th>Voating Classifier, Ensembled Models: ('dt', 'knn', 'rf', 'xgb', 'gdb', 'lgbm', 'adb', 'lr', 'rf')</th>
   <th>0.863</th>
    <th>0.828</th>
    <th>0.863</th>
    <th>0.823</th>
    <th><img width="1080" height="960" alt="image" src="https://github.com/user-attachments/assets/c13fc6ef-96ef-4693-a791-c02efd494a11" />
</th>
  </tr>
  
  <tr>
    <th>Blending Classifier, Base Models: ('rf', 'xgb', 'dt', 'knn', 'lgbm', 'adb'), Meta Model : ('lr')</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>

  <tr>
    <th>Stacking Classifier, Base Models: ('rf', 'xgb', 'dt', 'knn', 'gdb', 'lgbm', 'adb'), Meta Model : ('lr')</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
</table>


# MultiLingual Data (Arabic, Spanish, English, Dutch)
## MultiLingual BERT (PEFT) QLoRA, Model: Microsoft BERT (Encoder)

### Evaluation Dataset Scores

| Metric     | Value        |
|------------|--------------|
| Accuracy   | -            |
| F1 Score   | -            |
| Precision  |-             |
| Recall     | -            |

## XLM RoBERTa (PEFT) QLoRA, Model: Microsoft BERT (Encoder)

### Evaluation Dataset Scores

| Metric     | Value        |
|------------|--------------|
| Accuracy   | -            |
| F1 Score   | -            |
| Precision  |-             |
| Recall     | -            |


## This result has already surpassed the best scores from CLEF 2024, as seen [here](https://checkthat.gitlab.io/clef2024/task1/).

[checkThat](https://checkthat.gitlab.io/clef2024/task1/)
<img width="506" alt="image" src="https://github.com/user-attachments/assets/e5e0c9f1-8f0a-485d-9d55-04adc89a4833" />

