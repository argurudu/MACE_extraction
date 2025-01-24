import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize

'''
Read the chosen dataset below.
'''
df = pd.read_csv('mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/discharge.csv.gz')


'''
Sentence Tokenization:
Split all reports into sentence tokens that are separated by a period, new line, or semicolon. Store these tokens in a new column. 
'''

allSentences = []
for text in df[r'text']:
    sentence_tokens = sent_tokenize(text)
    all_sentences.append(sentence_tokens)
df[r'sentence_tokens'] = allSentences


