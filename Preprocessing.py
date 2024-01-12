import pandas as pd
import fasttext
import inflect
import re

# Code can take a long time to run (~1.5 hours)
# For the results of the research paper, you can just run main.py
# (and EDA.py for summary statistics of data) using the preprocessed_data.csv
# file in the data folder.

# Load in the original data
data = pd.read_csv("data/nationality.csv")

# Load the pre-trained language identification model
language_model = fasttext.load_model('data/lid.176.bin')  # Download from https://fasttext.cc/docs/en/language-identification.html

# Function to check the language of the (majority of the) text with the corresponding confidence
def is_english_fasttext(text):
    ''' Predicts for a certain text the language that
    text is in and gives a confidence rate for that prediction.'''
    predictions = language_model.predict(text, k=1)
    lang_label, conf = predictions[0][0], predictions[1][0]
    return lang_label, conf

# List to keep track of the indices that should be removed
idx_to_remove = []

# For loop to find the users whose posts are not English (most of the time)
# with 95% confidence in order to remove those from the dataset, because
# otherwise the tf idf model will train on the non-English words to predict
# non-English-speaking nationalities.
for idx, test in enumerate(data['post']):
    # Retrieve the corresponding language and confidence for a certain user's posts
    lang, conf = is_english_fasttext(test)

    # Check if the label is not English or if the label is English, but
    # not with enough confidence (>95%) and add them to the remove list.
    if lang != '__label__en' or (lang == '__label__en' and conf < 0.95):
        # print(idx, lang, conf)
        # print(test)
        idx_to_remove.append(idx)

# Remove the entries of the users with (too many) non-English posts from the dataset
data_filtered = data.drop(index=idx_to_remove).reset_index(drop=True)
print(data_filtered)


# Now we will delete all place names (such as countries, regions, cities)
# and their demonyms (and the plurals of those place names and demonyms)

# This function will delete all occurences of those places in the strings
def remove_words_from_string(word_list, input_string):
    ''' Removes particular words that are in a word_list
     from a string. We use this to delete place names and
     their demonyms from the posts.'''

    # Create a regular expression pattern from the list of words
    pattern = r'\b(?:' + '|'.join(re.escape(word) for word in word_list) + r')\b'

    # Use re.sub to replace the matched words with an empty string
    result_string = re.sub(pattern, '', input_string)

    return result_string

# Create an inflect engine (this is to create plurals of words)
p = inflect.engine()

# Load in the already partially filtered data
data = data_filtered.copy()

# Only get the entries of users with a nationality that has more than 1000 entries
data = data.groupby('nationality').filter(lambda x: len(x) > 1000)

# Load in the file with place names and their demonyms
demonyms = pd.read_csv('data/demonyms.csv', header=None)
demonyms = demonyms.rename(columns={0: "Demonym", 1: "Place"})
demonym_list = demonyms['Demonym'].to_list()
place_list = demonyms['Place'].to_list()
all_singular_places = demonym_list + place_list

# Add the plurals of those places to the list and convert them all to lowercase
plural_places = [p.plural(place) for place in all_singular_places]
all_places = all_singular_places + plural_places
all_places_lowercase = [place.lower() for place in all_places]

# Some initializations
new_post_column = []
count = 0

# Loop over all posts in the dataset and remove the place names
# and their demonyms (and the plurals of those).
# This can take a lot of time to run (1.5 hours).
for post in data['post']:
    # Convert the post to lowercase and remove the places
    post_lowercase = post.lower()
    post_cleaned = remove_words_from_string(all_places_lowercase, post_lowercase)

    # Add the cleaned post to a list
    new_post_column.append(post_cleaned)

    # Check to see where we are at
    if count % 1000 == 0:
        print('We are at count: ', count)
    count += 1

# Replace the old post column with the cleaned one
data['post'] = new_post_column

# Save the filtered dataframe as a .csv file so this code does not have to be run every time.
data.to_csv('data/preprocessed_data.csv')