import pandas as pd
import re

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
    allSentences.append(re.split(r'\.|\n|;', text))
df[r'sentence_tokens'] = allSentences


