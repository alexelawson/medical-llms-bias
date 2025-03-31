# Medical Bias in LLMS Investigation: Topics in Artificial Intelligence - NLP COMMONSENSE-Final Project

**Results from First Analysis:**

- _**Visuals**_ from the first analysis can be found in the [Visuals](./Visuals) folder. 

- _**Results**_ .csv can be found in the [Results01](./Results01) folder.

- _**Python Scripts Used to Produce Analysis/Results**_ can be found in the [Data-Augmentation-Scripts](./Data-Augmentation-Scripts) folder. The prompter for GPT augmentation can also be found in this folder. 

**Analysis Pipeline Instructions:**

1. First, merge the augmented questions with original data:
```bash
python Data-Augmentation-Scripts/augmented-original-file-merge.py
```
Note: You may need to adjust file paths in the script before running.

2. Generate demographic analysis results:
```bash
python Data-Augmentation-Scripts/bydemographic.py
```
This will create various CSV files including demo_accuracy.csv in the Results01 folder.

Additional Analysis Scripts:
- `differences-by-topic.py`: Analyzes accuracy differences by subject/topic
- `more-analysis.py` and `more-plots.py`: Generates additional metrics and visualizations
- `all-differences.py`: Performs comprehensive difference analysis

**Next Steps:**

**Datasets** for second analysis can be found in the [Datasets](./Datasets) folder. 
_Within the datasets folder there are two files:_
  - `GPT3.5turbo-augmentedquestions02.csv`: the dataset with the augmented questions (this is the one for the second analysis with RAG/baseline). The questions are slightly different as they were augmented with GPT-3.5-turbo, so the intial baseline should be run again (without RAG)
  - `pharmacology_psychiatry_filtered_dataset.csv`: this is the original dataset that was given to GPT-3.5-turbo (so there are no augmented questions, just for reference).


Link to presentation: https://docs.google.com/presentation/d/14F-jxxvuvpPKG26IfFLFM-R-xLd_rX36bqbI558hxGQ/edit?usp=sharing 

