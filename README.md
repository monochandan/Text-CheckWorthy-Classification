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
  <td align="center">179</td>
  <td align="center">55</td>
</tr>
</Table>

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
    <th>0.635</th>
    <th>0.788</th>
    <th>0.635</th>
    <th>0.663</th>
    <th><img width="1080" height="960" alt="Image" src="https://github.com/user-attachments/assets/abe9f4aa-279e-478b-a8b4-64451ca11709" /></th>
  </tr>
  <tr>
    <th>KNN Classifier</th>
    <th>0.768</th>
    <th>0.732</th>
    <th>0.768</th>
    <th>0.736</th>
    <th><img width="1080" height="960" alt="Image" src="https://github.com/user-attachments/assets/d84bc04b-8d16-4285-8ea7-ca9bde4eb065" /></th>
  </tr>

  
   <tr>
    <th>XGB Classifier</th>
    <th>0.957</th>
    <th>0.957</th>
    <th>0.957</th>
    <th>0.956</th>
     <th><img width="1080" height="960" alt="Image" src="https://github.com/user-attachments/assets/b36ab90e-4618-43e5-9980-1bb5888b36e0" /></th>
  </tr>

  <tr>
    <th>Random Forest Classifier</th>
    <th>0.825</th>
    <th>0.858</th>
    <th>0.825</th>
    <th>0.834</th>
    <th><img width="1080" height="960" alt="image" src="https://github.com/user-attachments/assets/13180b87-ca95-4964-8b12-f5b5d5d9f748" />
</th>
  </tr>

  <tr>
    <th>Gradient Boosting Classifier</th>
    <th>0.942</th>
    <th>0.942</th>
    <th>0.942</th>
    <th>0.940</th>
    <th><img width="1080" height="960" alt="Image" src="https://github.com/user-attachments/assets/6de40c39-6351-465c-b311-5a8b6c8aaf85" /></th>
  </tr>

   <tr>
    <th>Light GBM Classifier</th>
    <th>0.942</th>
    <th>0.943</th>
    <th>0.942</th>
    <th>0.941</th>
    <th><img width="1080" height="960" alt="Image" src="https://github.com/user-attachments/assets/73e97d0d-efef-4d30-99ef-0ab554e4fcc6" /></th>
  </tr>

  <tr>
    <th>Ada Boost Classifier</th>
    <th>0.908</th>
    <th>0.915</th>
    <th>0.908</th>
    <th>0.901</th>
    <th><img width="1080" height="960" alt="Image" src="https://github.com/user-attachments/assets/386a8631-a420-43a7-a86f-08385e62e1d2" /></th>
  </tr>

  <tr>
    <th>Logistic Regression</th>
    <th>0.891</th>
    <th>0.915</th>
    <th>0.891</th>
    <th>0.896</th>
    <th><img width="1080" height="960" alt="Image" src="https://github.com/user-attachments/assets/0b3b3acd-a115-477e-8a89-3617be74d8ad" /></th>
  </tr>

  <tr>
    <th>Voating Classifier, Ensembled Models: ('dt', 'knn', 'rf', 'xgb')</th>
    <th>0.934</th>
    <th>0.934</th>
    <th>0.934</th>
    <th>0.931</th>
    <th><img width="1080" height="960" alt="Image" src="https://github.com/user-attachments/assets/545293c9-99ae-4a0d-b262-27d4d02f7b4c" /></th>
  </tr>

  <tr>
    <th>Voating Classifier, Ensembled Models: ('rf', 'xgb', 'dt', 'knn', 'lgbm', 'adb')</th>
    <th>0.943</th>
    <th>0.944</th>
    <th>0.943</th>
    <th>0.942</th>
    <th><img width="1080" height="960" alt="image" src="https://github.com/user-attachments/assets/c71c97ac-314f-4225-9317-01cfe900f10a" />
</th>
  </tr>

  <tr>
    <th>Voating Classifier, Ensembled Models: ('gdb', 'lgbm', 'adb')</th>
    <th>0.946</th>
    <th>0.947</th>
    <th>0.946</th>
    <th>0.945</th>
    <th><img width="1080" height="960" alt="image" src="https://github.com/user-attachments/assets/ca5d5eea-840c-4145-8090-53de441ca2d8" />
</th>
  </tr>

  <tr>
    <th>Voating Classifier, Ensembled Models: ('dt', 'knn', 'rf', 'xgb', 'adb')</th>
    <th>0.934</th>
    <th>0.936</th>
    <th>0.934</th>
    <th>0.931</th>
    <th><img width="1080" height="960" alt="image" src="https://github.com/user-attachments/assets/02658698-a545-4f42-861f-345fdaf9a6d9" />
</th>
  </tr>


  <tr>
    <th>Voating Classifier, Ensembled Models: ('rf', 'xgb', 'dt')</th>
    <th>0.953</th>
    <th>0.953</th>
    <th>0.953</th>
    <th>0.953</th>
    <th><img width="1080" height="960" alt="image" src="https://github.com/user-attachments/assets/4458f702-f505-4f45-9ad3-5dee2a002be9" />
</th>
  </tr>

  <tr>
    <th>Voating Classifier, Ensembled Models: ('xgb', 'lgbm', 'adb', 'lr')</th>
    <th>0.957</th>
    <th>0.957</th>
    <th>0.957</th>
    <th>0.957</th>
    <th><img width="1080" height="960" alt="image" src="https://github.com/user-attachments/assets/f2965f37-4d3e-43ff-868a-4d55c321467d" />
</th>
  </tr>

  <tr>
    <th>Voating Classifier, Ensembled Models: ('dt', 'knn', 'rf', 'xgb', 'gdb', 'lgbm', 'adb', 'lr', 'rf')</th>
    <th>0.999</th>
    <th>0.999</th>
    <th>0.999</th>
    <th>0.999</th>
    <th></th>
  </tr>
  
  <tr>
    <th>Blending Classifier, Base Models: ('rf', 'xgb', 'dt', 'knn', 'lgbm', 'adb'), Meta Model : ('lr')</th>
    <th>0.624</th>
    <th>0.389</th>
    <th>0.624</th>
    <th>0.479</th>
    <th></th>
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





### Parameter Efficient Fine Tuning (PEFT) QLoRA, Model: Microsoft BERT (Encoder)

### Evaluation Dataset Scores

| Metric     | Value        |
|------------|--------------|
| Accuracy   | 0.9176       |
| F1 Score   | 0.8399       |
| Precision  | 0.7611       |
| Recall     | 0.9370       |


<!-- ############################################################################################################################--->

# Arabic Data

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
# Spanish Data

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
  <td align="center"> 704</td>
</tr>

<tr>
  <td>Dev Test Data (Model Test)</td>
  <td align="center">4491</td>
  <td align="center"> 509</td>
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
    <th>Decision Tree Classifier</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>0.911</th>
  </tr>
  <tr>
    <th>KNN Classifier</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>

  
   <tr>
    <th>XGB Classifier</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>

  <tr>
    <th>Random Forest Classifier</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>

  <tr>
    <th>Gradient Boosting Classifier</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>

   <tr>
    <th>Light GBM Classifier</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>

  <tr>
    <th>Ada Boost Classifier</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>

  <tr>
    <th>Logistic Regression</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
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
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>

  <tr>
    <th>Voating Classifier, Ensembled Models: ('dt', 'knn', 'rf', 'xgb', 'adb')</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>


  <tr>
    <th>Voating Classifier, Ensembled Models: ('rf', 'xgb', 'dt')</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>

  <tr>
    <th>Voating Classifier, Ensembled Models: ('xgb', 'lgbm', 'adb', 'lr')</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>

  <tr>
    <th>Voating Classifier, Ensembled Models: ('dt', 'knn', 'rf', 'xgb', 'gdb', 'lgbm', 'adb', 'lr', 'rf')</th>
   <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
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


# MultiLingual Data (Arabic, Spanish, English)
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

