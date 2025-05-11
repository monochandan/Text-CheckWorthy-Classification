Check Worthiness Estimetion of Text Data

This project addresses the growing need to identify claims worth fact-checking, especially in the age of widespread misinformation. Focusing on English-language datasets from U.S. presidential debate transcripts, we apply a range of resampling methods to tackle data imbalance and explore multiple machine learning approaches—from traditional models to fine-tuned LLMs using memory-efficient techniques like QLoRA.

Key contributions include:

- How to handle imbalance text data

- Evaluation of linguistic, contextual, and semantic features.

- Ensemble strategies that significantly boost performance.

- Benchmarking on CLEF 2024's CheckThat! dataset and additional tweet data.

- Preliminary results show ensemble-based classical models outperform current CLEF 2024 LLM-based baselines.

- Fine-tuning encoder LLM Model with QLoRA also give performance compared to LoRA.


With the rapid rise of online misinformation, it's increasingly important to prioritize which claims are worth fact-checking. Check-worthiness estimation tackles this by classifying whether a statement—like a tweet or debate quote—merits verification. However, challenges such as subjectivity, data imbalance, and linguistic ambiguity make this task difficult.

While recent benchmarks like CheckThat! Lab at CLEF 2024 have seen dominance from transformer-based and LLM-based models (e.g., RoBERTa, GPT-4, LLaMA2), these often demand high computational resources. This project explores whether ensemble-based traditional ML models, supported by resampling techniques, can remain competitive. We also experiment with QLoRA for memory-efficient fine-tuning of large models, offering a practical alternative to resource-heavy approaches.






<div style="display: flex; flex-wrap: wrap; gap: 20px;"> <div style="flex: 1; min-width: 300px;">
🔀 Voting Classifier (7 Models: RF, XGB, DT, KNN, GDB, LGBM, ADB)
Metric	Value
Accuracy	0.827
Precision	0.821
Recall	0.827
F1 Score	0.823

</div> <div style="flex: 1; min-width: 300px;">
🔀 Voting Classifier (3 Models: GDB, LGBM, ADB)
Metric	Value
Accuracy	0.809
Precision	0.806
Recall	0.809
F1 Score	0.808

</div> <div style="flex: 1; min-width: 300px;">
🔀 Voting Classifier (5 Models: DT, KNN, RF, XGB, ADB)
Metric	Value
Accuracy	0.801
Precision	0.793
Recall	0.801
F1 Score	0.796

</div> <div style="flex: 1; min-width: 300px;">
🌲 Random Forest Classifier (RFC)
Metric	Value
Accuracy	0.745
Precision	0.728
Recall	0.745
F1 Score	0.734

</div> <div style="flex: 1; min-width: 300px;">
🌳 Decision Tree Classifier (DT)
Metric	Value
Accuracy	0.721
Precision	0.729
Recall	0.721
F1 Score	0.725

</div> <div style="flex: 1; min-width: 300px;">
🤖 K-Nearest Neighbors (KNN)
Metric	Value
Accuracy	0.674
Precision	0.666
Recall	0.674
F1 Score	0.670

</div> <div style="flex: 1; min-width: 300px;">
🚀 XGBoost Classifier (XGB)
Metric	Value
Accuracy	0.795
Precision	0.805
Recall	0.795
F1 Score	0.799

</div> </div>
