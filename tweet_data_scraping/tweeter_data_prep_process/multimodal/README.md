# Working on multimodal data set: https://aclanthology.org/2022.findings-naacl.72.pdf

## Initial process:
1. Image caption and OCR text and Image were used as features.
2. Used [CLIP](https://huggingface.co/docs/transformers/v5.5.0/en/model_doc/clip#transformers.CLIPProcessor) to create embeddings.
3. Only 2 embeddings (image embeddings, text embeddings) used as features.
4. Train data on classical models, [votinclassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html), [stackingclassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html#)
   with multiple different setups from [ensemble.py](https://github.com/monochandan/Text-CheckWorthy-Classification/blob/main/Ensembling_techniques/Ensemble.py)
5. models [parameters](https://github.com/monochandan/Text-CheckWorthy-Classification/blob/main/hp-tuning/hp-tuning.ipynb) tuned on the [text-data](http://clef2024.clef-initiative.eu/index.php?page=Pages/lab_pages/checkthat.html)


## Result:
```
VOATING CLASSIFIER:
Classifiers used: ['dt', 'knn', 'rf', 'xgb']
Accuracy: 0.787
Precision: 0.791
Recall: 0.787
F1 Score: 0.776

Classifiers used: ['rf', 'xgb', 'dt', 'knn', 'lgbm', 'adb']
Accuracy: 0.780
Precision: 0.800
Recall: 0.780
F1 Score: 0.761

Classifiers used: ['gdb', 'lgbm', 'adb']
Accuracy: 0.772
Precision: 0.801
Recall: 0.772
F1 Score: 0.748

Classifiers used: ['dt', 'knn', 'rf', 'xgb', 'adb']
Accuracy: 0.787
Precision: 0.793
Recall: 0.787
F1 Score: 0.775

Classifiers used: ['rf', 'xgb', 'dt']
Accuracy: 0.772
Precision: 0.777
Recall: 0.772
F1 Score: 0.759

Classifiers used: ['xgb', 'lgbm', 'adb', 'lr']
Accuracy: 0.774
Precision: 0.790
Recall: 0.774
F1 Score: 0.757

Classifiers used: ['dt', 'knn', 'rf', 'xgb', 'gdb', 'lgbm', 'adb', 'lr', 'rf']
Accuracy: 0.787
Precision: 0.806
Recall: 0.787
F1 Score: 0.770

STACKING CLASSIFIER:

Classifiers Used ['dt', 'knn', 'rf', 'xgb']
Accuracy: 0.770
Precision: 0.786
Recall: 0.770
F1 Score: 0.752

Classifiers Used ['rf', 'xgb', 'dt', 'knn', 'lgbm', 'adb']
Accuracy: 0.770
Precision: 0.789
Recall: 0.770
F1 Score: 0.751

Classifiers Used ['gdb', 'lgbm', 'adb']
Accuracy: 0.769
Precision: 0.797
Recall: 0.769
F1 Score: 0.745

Classifiers Used ['dt', 'knn', 'rf', 'xgb', 'adb']
Accuracy: 0.768
Precision: 0.783
Recall: 0.768
F1 Score: 0.749

Classifiers Used ['rf', 'xgb', 'dt']
Accuracy: 0.776
Precision: 0.794
Recall: 0.776
F1 Score: 0.757

Classifiers Used ['xgb', 'lgbm', 'adb', 'lr']
Accuracy: 0.769
Precision: 0.787
Recall: 0.769
F1 Score: 0.749

Classifiers Used ['dt', 'knn', 'rf', 'xgb', 'gdb', 'lgbm', 'adb', 'lr', 'rf']

```
   
