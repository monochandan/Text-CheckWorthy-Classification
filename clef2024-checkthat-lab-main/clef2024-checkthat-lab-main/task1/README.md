# Task 1: Check-worthiness Estimation in Text

The aim of this task is to determine whether a claim in a tweet and/or transcriptions is worth fact-checking. Typical approaches to make that decision require to either resort to the judgments of professional fact-checkers or to human annotators to answer several auxiliary questions such as “does it contain a verifiable factual claim?”, and “is it harmful?”, before deciding on the final check-worthiness label https://aclanthology.org/2021.findings-emnlp.56.pdf.

This year, we are offering multi-genre data: the tweets and/or transcriptions should be judged based solely on the text. 

Along with the task, we release train and dev sets in Arabic, English, Dutch and Spanish. However, the official test set will be limited to Arabic, English, and Dutch. The Spanish dataset can be used for training.


__Table of contents:__

- [Dataset](#dataset)
- [Evaluation](#evaluation)
- [Submission Guidelines](#submission-guidelines)
- [List of Versions](#list-of-versions)
- [Contents of the Directory](#contents-of-the-directory)
- [File Format](#file-format)
	- [Check-Worthiness of multigenre content](#check-worthiness-of-multigenre-content)
		- [Input Data Format - Tweets](#input-data-format-tweets)
		- [Input Data Format - Political Debates](#input-data-format-political-debates)
	- [Output Data Format](#output-data-format)
- [Format Checkers](#format-checkers)
- [Scorers](#scorers)
- [Baselines](#baselines)
- [Credits](#credits)

## Dataset
Data for all languages are available [here](https://gitlab.com/checkthat_lab/clef2024-checkthat-lab/-/tree/main/task1/data)

Each instance is composed of only text, which could come from a tweet, the transcription of a debate or the transcription of speech.  


## Evaluation
This is a binary classification task. The official evaluation metric is F_1 over the positive class.


## Submission Guidelines:
- Make sure that you create one account for each team, and submit runs through one account only.
- The last file submitted to the leaderboard will be considered as the final submission.
- Name of the output file has to be `task1_lang.tsv` with `.tsv` extension (e.g., task1_arabic.tsv); otherwise, you will get an error on the leaderboard. Three languages are possible (Arabic, English, and Dutch).
- You have to zip the tsv, `zip task1_arabic.zip task1_arabic.tsv` and submit it through the codalab page.
- It is required to submit the team name and **method description** for each submission. **Your team name here must EXACTLY match that used during CLEF registration.**
- You are allowed to submit max 200 submissions per day.
- We will keep the leaderboard private till the end of the submission period, hence, results will not be available upon submission. All results will be available after the evaluation period.

<!-- **Please submit your results on test data here: https://codalab.lisn.upsaclay.fr/competitions/12936** -->


## List of Versions

* __[2024/03/19]__
  - Details of Task 1 is updated.

* __[2024/04/24]__
  - Details of Task 1 is updated.
  - Submission guidelines added.


## Contents of the Directory
* Main folder: [data](./data)
  	This directory contains files for all languages.

* Main folder: [baselines](./baselines)<br/>
	Contains scripts provided for baseline models of the tasks.
* Main folder: [format_checker](./format_checker)<br/>
	Contains scripts provided to check format of the submission file.
* Main folder: [scorer](./scorer)<br/>
	Contains scripts provided to score output of the model when provided with label (i.e., dev and dev-test sets).

* [README.md](./README.md) <br/>
	This file!


## File Format


### Check-Worthiness of multigenre content


#### Input Data Format (Tweets)
For **Arabic**, **Spanish** and **Dutch** we use the same data format in the train, dev and dev-test files. Each file is TAB seperated (TSV file) containing the tweets and their labels. The text encoding is UTF-8. Each row in the file has the following format:

> tweet_id <TAB> tweet_url <TAB> tweet_text <TAB> class_label

Where: <br>
* tweet_id: Tweet ID for a given tweet given by Twitter <br/>
* tweet_url: URL to the given tweet <br/>
* tweet_text: content of the tweet <br/>
* class_label: *Yes* and *No*


**Examples:**
> 1235648554338791427	https://twitter.com/A6Asap/status/1235648554338791427	COVID-19 health advice⚠️ https://t.co/XsSAo52Smu	No<br/>
> 1235287380292235264	https://twitter.com/ItsCeliaAu/status/1235287380292235264	There's not a single confirmed case of an Asian infected in NYC. Stop discriminating cause the virus definitely doesn't. #racist #coronavirus https://t.co/Wt1NPOuQdy	Yes<br/>
> 1236020820947931136	https://twitter.com/ddale8/status/1236020820947931136	Epidemiologist Marc Lipsitch, director of Harvard's Center for Communicable Disease Dynamics: “In the US it is the opposite of contained.' https://t.co/IPAPagz4Vs	Yes<br/>
> ... <br/>

Note that the gold labels for the task are the ones in the *class_label* column.



#### Input Data Format (Political debates)
For **English** we use the same data format in the train, dev and dev-test files. Each file is TAB seperated (TSV file) containing the sentences and their labels. The text encoding is UTF-8. Each row in the file has the following format:

> Sentence_id <TAB> Text <TAB> class_label

Where: <br>
* Sentence_id: sentence id for a given political debate <br/>
* Text: sentence's text <br/>
* class_label: *Yes* and *No*


**Examples:**
> 30313	And so I know that this campaign has caused some questioning and worries on the part of many leaders across the globe.	No<br/>
> 19099	"Now, let's balance the budget and protect Medicare, Medicaid, education and the environment."	No<br/>
> 33964	I'd like to mention one thing.	No<br/>
> ... <br/>

Note that the gold labels for the task are the ones in the *class_label* column.


### Output Data Format
For all languages (**Arabic**, **English**, and **Dutch**) the submission files format is the same.

The expected results file is a list of tweets/transcriptions with the predicted class label.

The file header should strictly be as follows:

> **id <TAB> class_label <TAB> run_id**

Each row contains three TAB separated fields:

> tweet_id or id <TAB> class_label <TAB> run_id

Where: <br>
* tweet_id or id: Tweet ID or sentence id for a given tweet given by Twitter or coming from political debates given in the test dataset file. <br/>
* class_label: Predicted class label for the tweet. <br/>
* run_id: String identifier used by participants. <br/>

Example:
> 1235648554338791427	No  Model_1<br/>
> 1235287380292235264	Yes  Model_1<br/>
> 1236020820947931136	No  Model_1<br/>
> 30313	No  Model_1<br/>
> ... <br/>


## Format Checkers

The checker for the task is located in the [format_checker](./format_checker) module of the project.
To launch the checker script you need to install packages dependencies found in [requirements.txt](./requirements.txt) using the following:
> pip3 install -r requirements.txt <br/>

The format checker verifies that your generated results files complies with the expected format.
To launch it run:

> python3 format_checker/task_1.py --pred-files-path <path_to_result_file_1 path_to_result_file_2 ... path_to_result_file_n> <br/>

`--pred-files-path` is to be followed by a single string that contains a space separated list of one or more file paths.

__<path_to_result_file_n>__ is the path to the corresponding file with participants' predictions, which must follow the format, described in the [Output Data Format](#output-data-format) section.

Note that the checker can not verify whether the prediction files you submit contain all tweets, because it does not have access to the corresponding gold file.


## Scorers

The scorer for the task is located in the [scorer](./scorer) module of the project.
To launch the script you need to install packages dependencies found in [requirements.txt](./requirements.txt) using the following:
> pip3 install -r requirements.txt <br/>

Launch the scorer as follows:
> python3 scorer/task_1.py --gold-file-path=<path_gold_file> --pred-file-path=<prediction_file> --lang=<language> <br/>

`--lang` expects one of four options **arabic** or **english** or **dutch** or **spanish** to indicate the language for which to score the predictions file.

The scorer invokes the format checker for the task to verify the output is properly shaped.
It also handles checking if the provided predictions file contains all tweets from the gold one.


## Baselines

The [baselines](./baselines) module currently contains a majority, random and a simple n-gram baseline.

**Baseline Results for Task 1 on Dev_Test**
|Model|task-1--Arabic|task-1--English|task-1--Dutch|task-1--Spanish|
|:----|:----|:----|:----|:----|
|Majority Baseline|0.000|0.000|0.000|0.000|
|Random Baseline |0.625|0.462|0.482|0.172|
|n-gram Baseline|0.369|0.599|0.510|0.391|


To launch the baseline script you need to install packages dependencies found in [requirements.txt](./requirements.txt) using the following:
> pip3 install -r requirements.txt <br/>

To launch the baseline script run the following:
> python3 baselines/task_1.py --train-file-path=<path_to_your_training_data> --dev-file-path=<path_to_your_test_data_to_be_evaluated> --lang=<language_of_the_task><br/>
```
python3 baselines/task_1.py --train-file-path=data/CT24_checkworthy_arabic/CT24_checkworthy_arabic_train.tsv --dev-file-path=data/CT24_checkworthy_arabic/CT24_checkworthy_arabic_dev-test.tsv -l arabic
```


All baselines will be trained on the training dataset and the performance of the model is evaluated on the dev-test.

## Credits
Please find it on the task website: https://checkthat.gitlab.io/clef2024/task1/

<!-- Contact:   clef-factcheck@googlegroups.com -->
