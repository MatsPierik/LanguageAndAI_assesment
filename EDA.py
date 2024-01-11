import pandas as pd
import matplotlib.pyplot as plt
from langid.langid import LanguageIdentifier, model

pd.set_option('display.max_columns', None)
# df = pd.read_csv('data/nationality.csv')
# filtered_df = df.groupby('nationality').filter(lambda x: len(x) > 2000).reset_index(drop=True)
# print(filtered_df)
# num_posts = len(filtered_df)
# num_nationalities = len(filtered_df["nationality"].unique())
# num_authors = len(filtered_df["auhtor_ID"].unique())
#
# print(f"Number of Posts: {num_posts}")
# print(f'Number of Nationalities: {num_nationalities}')
# print(f'Number of Authors: {num_authors}')
#
# # Distribution of posts per nationality
# nation_post_count = filtered_df.groupby('nationality')[['post']].count()
# nation_post_count = nation_post_count.sort_values('post', ascending=False)
# nation_post_count.rename(index={'United Kingdom': 'UK', 'The Netherlands' : 'Holland'}, inplace=True)
#
# fig, ax = plt.subplots()
# ax.bar(nation_post_count.index, nation_post_count['post'])
# ax.set_title('Distribution of Posts per Nationality')
# ax.set_ylabel('Number of Posts')
# ax.set_xlabel('Nationality')
# plt.xticks(rotation=45, fontsize=8)
# plt.show()

# # To create a list with country and place names
# demonyms = pd.read_csv('data/demonyms.csv', header=None)
# demonyms = demonyms.rename(columns={0: "Demonym", 1: "Place"})
# demonym_list = demonyms['Demonym'].to_list()
# place_list = demonyms['Place'].to_list()
# all_places = demonym_list + place_list
# print(all_places)
# print(len(all_places))

def is_english(text):
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    lang, confidence = identifier.classify(text)
    # You can adjust the confidence threshold based on your needs
    return lang, confidence

# Example usage:
data = pd.read_csv("data/nationality.csv")
idx_to_remove = []
for idx, test in enumerate(data['post']):
    lang, conf = is_english(test)
    if lang != 'en' or (lang == 'en' and conf < 0.8):
        idx_to_remove.append(idx)
data_filtered = data.drop(index=idx_to_remove)