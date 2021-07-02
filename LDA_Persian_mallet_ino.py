#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


import pandas as pd
import numpy as np
import os
import json
import re
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


with open(os.path.join("C:/Users/Ваня/Downloads/Telegram Desktop/BBC", "result.json") ,'r', encoding='utf-8') as a:
    texts1 = json.load(a)
with open(os.path.join("C:/Users/Ваня/Downloads/Telegram Desktop/VOA2", "result.json") ,'r', encoding='utf-8') as b:
    texts2 = json.load(b)
with open(os.path.join("C:/Users/Ваня/Downloads/Telegram Desktop/Radio Farda", "result.json") ,'r', encoding='utf-8') as c:
    texts3 = json.load(c)
pprint(texts3)

tele_data = []
for text in texts1['messages']:
    tele_data.append(text['text'])    
#print(len(tele_data))
for text in texts2['messages']:
    tele_data.append(text['text'])
#print(len(tele_data))
for text in texts3['messages']:
    tele_data.append(text['text'])
print(len(tele_data))


# In[3]:


def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub('', string)


# In[5]:


txt_sub0 = [remove_emoji(str(text)) for text in tele_data] # no emoji


# In[6]:


txt_sub1 = [re.sub("\\\\u200c", "A", str(text)) for text in txt_sub0]
txt_sub2 = [re.sub("\\\\n",  " ", str(text)) for text in txt_sub1]


# In[8]:


#from __future__ import unicode_literals
from hazm import *

normalizer = Normalizer()

#from hazm import sent_tokenize, word_tokenize, Normalizer, Lemmatizer
from parsivar import Normalizer, Tokenizer, FindStems

norm = Normalizer()
tok = Tokenizer()
stem = FindStems()
bow = [tok.tokenize_words(text) for text in txt_sub2]

alpha_num = [[word for word in sent if word.isalpha()] for sent in bow]
alpha_num1 = [[re.sub("A", "‌", word) for word in text] for text in alpha_num]
stems = [[stem.convert_to_stem(word) for word in text] for text in alpha_num1]
stems_all = [[re.sub("\&\w+", "", word) for word in text] for text in stems]
print(stems_all[:10])


# In[9]:


import gensim

bigram = gensim.models.Phrases(stems_all, min_count=20, threshold=50) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[stems_all], threshold=25)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
words_bi = [bigram_mod[doc] for doc in stems_all]
words_tri = [trigram_mod[bigram_mod[doc]] for doc in words_bi]
print(words_tri[:10])


# In[10]:


with open("D:/Uchyoba_gumanitaristika/Comp/persian", 'r', encoding='utf-8') as f:
    persian_words = f.read()


# In[29]:


pre_corpus = [x for x in words_tri if x != []]
corpus0 = [[word.lower() for word in text if word not in persian_words] for text in pre_corpus]
p_words = ['اش', 'ها' ,'می', 'های', 'ای', "آغاز", "پایان", "بر_اساس" , "بهمن", "هزار_نفر", "بر_اثر", "روز_شنبه", "ماه", "عکس_های_روز", "ممکن_اس", "میلیون_نفر"]
mycorpus = [[word for word in text if word not in p_words] for text in corpus0]
#corpus1 = [re.sub(r"\w+\_\w+", "", ), ]
#corpus_corrected = [" ".join(x) for x in corpus]

#import random
#random.shuffle(corpus_corrected)


# In[14]:


import gensim.corpora as corpora
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel


# In[31]:


id2word = Dictionary(mycorpus) # отображение между нормализованными словами и их целочисленными идентификаторами
print(len(id2word))

id2word.filter_extremes(no_below=5, no_above=0.1) # удаляем экстремально частые и экстремально редкие слова и словосочетания 
print(len(id2word))

texts = mycorpus
corpus = [id2word.doc2bow(text) for text in texts] # для каждого обращения создаем набор пар id-слово - частота

print(id2word[10])
print(id2word[100])
print(corpus[10])


# In[12]:


#!wget http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip


# In[13]:


#!unzip -q mallet-2.0.8.zip


# In[25]:


os.environ['MALLET_HOME'] = 'C:\\mallet-2.0.8'
mallet_path = 'C:\\mallet-2.0.8\\bin\\mallet.bat'


# In[27]:


mallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=12, id2word=id2word)

for idx, topic in mallet.show_topics(num_topics=12
                                        , formatted=False
                                        , num_words=30):
    print('Topic: {} \nWords: {}'.format(idx, '|'.join([w[0] for w in topic])))


# In[ ]:





# In[32]:


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=2):

    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v', topn=10)
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# In[35]:


get_ipython().run_cell_magic('time', '', '\nmodel_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=texts, start=2, limit=35, step=5)')


# In[36]:


# Show graph
limit=35; start=2; step=5;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


# In[37]:


for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 2))


# In[38]:


optimal_model = model_list[3]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10))


# In[39]:


optimal_model.show_topics(formatted=False, num_topics=-1)


# In[40]:


# Get topic weights
topic_weights = []
for i, row_list in enumerate(optimal_model[corpus]):
    topic_weights.append([w for i, w in row_list])


# In[41]:


# Array of topic weights    
arr = pd.DataFrame(topic_weights).fillna(0).values


# In[42]:


# Keep the well separated points (optional)
#arr = arr[np.amax(arr, axis=1) > 0.35]

# Dominant topic number in each doc
topic_num = np.argmax(arr, axis=1)


# In[43]:


# tSNE Dimension Reduction
tsne_model = TSNE(n_components=2, verbose=1, random_state=120, angle=.99, init='pca')
tsne_lda = tsne_model.fit_transform(arr)


# In[44]:


tsne_lda = pd.DataFrame(tsne_lda)


# In[45]:


tsne_lda['topic'] = topic_num


# In[46]:


plt.figure(figsize=(15,15))
sns.scatterplot(data=tsne_lda, x=tsne_lda[0], y=tsne_lda[1], hue="topic", palette=sns.color_palette("hls", 5)+sns.color_palette("Paired"), markers=True) #sns.color_palette("hls", 5)
plt.savefig('example.png')


# In[45]:





# In[ ]:




