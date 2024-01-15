# Language and AI: Classification of Reddit users' nationalities using BoW vs. Stylometry

This repository holds the source code for the assignment for Language and AI (JBC090).
We compared two models in a task of classifying the nationality of Reddit users based
on the posts they published on the platform. We were interested to see if a model based
on traditional Bag of Words (BoW) or based on stylometry features would perform better at
this task.

## What are the files in this directory?
* The first step is to collect all the data files. The file 'demonyms.csv' is available in
the data folder. However, the files 'nationality.csv', 'preprocessed_data.csv' and
'lid.176.bin' are all to big in size to add to this GitHub repository. Therefore, you have
to download those from a Google Drive (which link can be found in the data folder in the file
'other_data_links') and move those files to the data folder.
* Essentially, you can just run the 'main.py' file to obtain the results from the research
paper. Do note that this can take a long time to run (~2-3 hours). The file 
'preprocessed_data.csv', that is used as input data, is constructed in the 
'Preprocessing.py' file.
* The 'Preprocessing.py' file contains the code for the preprocessing of the data. First,
all entries of posts are deleted that are (for the majority) non-English (95% confidence).
This is done using the library fasttext and a pretrained model of theirs which can be 
downloaded from here: https://fasttext.cc/docs/en/language-identification.html.
Then, all the place names (countries, regions, cities etc.), their demonyms (e.g. Romanian
for Romania or Londoner for London) and the plural of those two categories (e.g. Romanians
or Londoners) are removed from the remaining posts. We did this by using the 'demonyms.csv'
file, which contained 2.144 places. We obtained this file from: 
https://github.com/knowitall/chunkedextractor/blob/master/src/main/resources/edu/knowitall/chunkedextractor/demonyms.csv
* The 'EDA.py' file contains some summary statistics of the dataset, which was used to 
write the '3. Data' segment of the research paper.
* The 'main.py' file contains the code for the comparison of the models. It contains the
majority baseline model, which just predicts the nationality that contains the most entries. 
Additionally, we have two models that use Term Frequency and Inverse Document Frequency 
(TF-IDF) as input: Naive Bayes (NB) and Support Vector Machine (SVM). However, we have seen
during the experiments that SVM performs better than NB, so we will mainly focus on that model.
Next, we also use an SVM model with stylometry features as input. And finally, we use 
a SVM model with TF-IDF as input and stylometry features as input.

## How to run the code?
* Essentially, you can just run the 'main.py' file to obtain the results from the research
paper. Do note that this can take a long time to run (~2-3 hours). The file 
'preprocessed_data.csv', that is used as input data, is constructed in the 
'Preprocessing.py' file.
* If you are wondering how we obtained the 'preprocessed_data.csv' file, you can run the 
'Preprocessing.py' file. This file takes the original 'nationality.csv' file as input (as 
received from the lecturer) and performs preprocessing tasks to clean the data. This file
also takes a long time to run (~1.5-2 hours).
* If you are wondering how we obtained the summary statistics in the '3. Data' segment of
the research paper, you can run the 'EDA.py' file. This does not take long to run.

## Packages needed to run the code
* The file requirements.txt contains all the packages used (and their versions) in the
code. You can install those packages by executing the following command in the command line:
'''
pip install -r requirements.txt
''' 
* Most packages used in the code are standard, such as pandas, numpy, sklearn, nltk, 
matplotlib, seaborn and regex.
* Two packages that are less known are the fasttext package, for which we installed version
0.9.2, and the inflect package, for which we installed version 7.0.0. Other versions will 
probably also work, but for these we are sure that the code works.
