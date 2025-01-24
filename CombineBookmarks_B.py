#!/usr/bin/env python
# coding: utf-8

# ## Créer un fichier favoris combiné et dédoublonné

# In[1]:


import os
import json
from bs4 import BeautifulSoup


# In[2]:


# Définir la fonction parser avec beautifulsoup
def parser_favoris(html_file):
    """Parser un fichier HTML de favoris."""
    with open(html_file, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'lxml')
    
    # Trouver tous les liens
    links = soup.find_all('a')
    
    favoris = []
    for link in links:
        href = link.get('href')
        if href:
            title = link.text
            favoris.append({'title': title, 'href': href})
    
    return favoris


# In[3]:


# Définir une fonction pour combiner les 2 listes de favoris, tout en supprimant les doublons d'url parmi eux
def combiner_favoris(favoris_pro, favoris_perso):
    """Combiner deux listes de favoris, en supprimant les doublons."""
    # Utiliser un dictionnaire pour supprimer les doublons
    favoris_dict = {}
    for favori in favoris_pro + favoris_perso:
        favoris_dict[favori['href']] = favori
    
    # Convertir le dictionnaire en liste
    favoris_combines = list(favoris_dict.values())
    
    return favoris_combines


# In[4]:


# Définir une fonction qui sauvegarde en json les favoris combinés
def sauvegarder_favoris(favoris, output_file):
    """Sauvegarder les favoris combinés dans un fichier JSON."""
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(favoris, file, indent=4, ensure_ascii=False)


# In[5]:


if __name__ == "__main__":
    favoris_pro_file = 'favoris_pro.html'
    favoris_perso_file = 'favoris_perso.html'
    output_file = 'favoris_combines.json'
    
    favoris_pro = parser_favoris(favoris_pro_file)
    favoris_perso = parser_favoris(favoris_perso_file)
    
    favoris_combines = combiner_favoris(favoris_pro, favoris_perso)
    
    sauvegarder_favoris(favoris_combines, output_file)
    
    print("Favoris combinés avec succès!")


# In[6]:


import pandas as pd
import re

# Charger le fichier JSON dans un DataFrame
df_bookmark = pd.read_json('favoris_combines.json')

# Supprimer les émojis et les caractères non ASCII
df_bookmark['title'] = df_bookmark['title'].apply(lambda x: re.sub(r'[^\x00-\x7F]+', '', x))


# In[7]:


# Information sur le DataFrame
df_bookmark.info()


# In[8]:


# Visualiser le DataFrame
df_bookmark.head(20)


# ## Machine Learning - clustering (DBSCAN) avec traitement NLP (SPACY)

# ### NLP avec SPACY

# In[76]:


import re
import unicodedata
from deep_translator import GoogleTranslator
import numpy as np
import spacy
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer


# In[10]:


# Chargement des données
df = pd.read_json('favoris_combines.json')


# In[11]:


# Prétraitement des données
def clean_text(text):
    # Enlever les Emojis
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Enlever les caractères non ASCII
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    # Enlever les caractères spéciaux
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remplacement des caractères spéciaux
    text = re.sub(r'[,\.-]', '', text)  # Remplacer ',' et '.' et '-' par ''
    return text


# In[12]:


# Traduction en français
def translate_text(text):
    translator = GoogleTranslator(source='auto', target='fr')
    translation = translator.translate(text)
    return translation


# In[13]:


# Finalisation prétraitement
def preprocess_title(title):
    # Nettoyage du texte
    title = clean_text(title)
    # Traduction en français si nécessaire
    title = translate_text(title) if title.isascii() else title
    # Mise en minuscule
    title = title.lower()
    return title


# In[28]:


df['title'] = df['title'].str.replace(',', '').str.replace('.', '').str.replace('-', '').str.replace('|', '')
df['title'] = df['title'].str.lower()
# df['title'] = df['title'].apply(lambda x: translate_text(x) if x.isascii() else x)


# In[32]:


df['title'].head(30)


# In[29]:


import spacy
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop


# In[65]:


final_stopwords_list = list(fr_stop) + list(en_stop)

len(final_stopwords_list)


# In[43]:


# Charger le modèle de langue français
nlp_fr = spacy.load("fr_dep_news_trf")


# In[66]:


# Définir le vectorizer avec les modèles de langage
tfidf_vectorizer = TfidfVectorizer(
    max_df=0.9999,
    # max_features=2000000,
    min_df=0.0009,
    stop_words=final_stopwords_list,
    use_idf=True,  
    ngram_range=(2,10)
)


# In[67]:


tfidf_vectorizer


# In[68]:


# Transformer les données textuelles en un vecteur numérique
X = tfidf_vectorizer.fit_transform(df_bookmark['title'])

X.shape


# In[69]:


type(X)


# In[70]:


print(X)


# ### Clustering avec DBSCAN

# In[85]:


dbscan = DBSCAN(eps=0.00005, min_samples=50)
dbscan.fit(X)


# In[86]:


len(dbscan.labels_)


# In[ ]:


# Génération des catégories et sous-catégories
categories = []
for i, cluster in enumerate(dbscan.labels_):
    category = df.iloc[i]['title']
    subcategories = []
    for j, subcluster in enumerate(dbscan.labels_):
        if subcluster == cluster:
            subcategory = df.iloc[j]['title']
            subcategories.append(subcategory)
    categories.append((category, subcategories))

# Affichage des catégories et sous-catégories
for category, subcategories in categories:
    print(f"Catégorie : {category}")
    for subcategory in subcategories:
        print(f"  - {subcategory}")


# In[ ]:




