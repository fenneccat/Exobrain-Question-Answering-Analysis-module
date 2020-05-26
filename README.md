# exobrain-query-analyzer-2019


## Query module

Code for analysing query. Add additional field to triviaQA json file to provide query analysis information.

This program needs `TriviaQA` dataset.
https://nlp.cs.washington.edu/triviaqa/

It updates train, text triviaQA data with additional information.

[Additional field]
- AnswerNERp: predicted NER type of answer for query
- WHtype: WH words in query
- QueryType: type of query [None; Counting; Quotes]
	- None: general method to make CG
	- Counting: How many question
		1. Change 'how many' to what and counting the number of answers in list
		2. Generate CG which has cardinal value as wildcard
	- Quotes: query which has explicit quotes
- AnswerType_d: detailed version of answertype
- AnswerType_ds: simple version of answertype
- Reformedq: reformulated question
	- WHword is changed as 'wildcard'
	- 'Quotes' query type has list of reformulated question
		1. WH word is chaged to 'wildcard'
		2. explicit quotes are changed to special token 'QUOTES'
		3. list of the original forms of quotes which is replaced as 'QUOTES'

### Run file

```
python query_analysis_v02.py
```
### Requirement
* SPACY (https://spacy.io/)
* NLTK
* tqdm


## Get LAT Features

Measuring similarity between answer type of given answer type and answer candidates.
Fine Grained Entity Typing model is used for get a type of answer (https://github.com/uwnlp/open_type)

### Run file
```
python get_LAT_feature.py release_model -lstm_type single -enhanced_mention -data_setup joint -add_crowd -multitask -mode test -reload_model_name release_model -eval_data crowd/test3.json -load
```
