import pandas as pd
from langid.langid import LanguageIdentifier, model

def is_english(text):
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    lang, confidence = identifier.classify(text)
    # You can adjust the confidence threshold based on your needs
    return lang, confidence

data = pd.read_csv("data/nationality.csv")
data = data.groupby('nationality').filter(lambda x: len(x) > 2000).reset_index(drop=True)
# data = data[:200]
# print(data)
idx_to_remove = []
for idx, test in enumerate(data['post']):
    lang, conf = is_english(test)
    if lang != 'en' or (lang == 'en' and conf < 0.95):
        print(idx, lang, conf)
        print(test)
        idx_to_remove.append(idx)
data_filtered = data.drop(index=idx_to_remove).reset_index(drop=True)
print(data_filtered)
# data_filtered.to_csv('data/preprocessed_data.csv')