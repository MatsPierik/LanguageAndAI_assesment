import pandas as pd
import matplotlib.pyplot as plt

# Load in original data
original_data = pd.read_csv('data/nationality.csv')

# Showcase some summary statistics over original dataset
or_num_posts = len(original_data)
or_num_nationalities = len(original_data["nationality"].unique())
or_num_authors = len(original_data["auhtor_ID"].unique())
print(f"Original Number of Posts: {or_num_posts}")
print(f'Original Number of Nationalities: {or_num_nationalities}')
print(f'Original Number of Authors: {or_num_authors}')

# Load in cleaned data
df = pd.read_csv('data/preprocessed_data.csv').drop(columns=['Unnamed: 0'])
df = df.groupby('nationality').filter(lambda x: len(x) > 1000)

# Showcase some summary statistics over cleaned dataset
num_posts = len(df)
num_nationalities = len(df["nationality"].unique())
num_authors = len(df["auhtor_ID"].unique())
print(f"Number of Posts after cleaning: {num_posts}")
print(f'Number of Nationalities after cleaning: {num_nationalities}')
print(f'Number of Authors after cleaning: {num_authors}')

# Find the number of entries per nationality for a bar plot
nation_post_count = df.groupby('nationality')[['post']].count()
nation_post_count = nation_post_count.sort_values('post', ascending=False)
nation_post_count.rename(index={'United Kingdom': 'UK', 'The Netherlands' : 'Holland'}, inplace=True)

# Create a bar plot that shows the distribution of entries per nationality
fig, ax = plt.subplots()
ax.bar(nation_post_count.index, nation_post_count['post'])
ax.set_title('Distribution of Entries per Nationality')
ax.set_ylabel('Number of Entries (Subsets of Posts)')
ax.set_xlabel('Nationality')
plt.xticks(rotation=45, fontsize=8)
plt.show()