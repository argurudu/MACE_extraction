import pandas as pd
import re

from sentence_tokenization import df

'''
Define the full set of MACE keywords and their synonyms, developed manually.
'''

keywords = [r"(myocardial|cardiac) infarct",
            r"(myocardial|coronary|cardiac) necrosis",
            r"heart (attack|infarct|rupture)",
            r"(attack|attacks|attacking) (heart|coronary)",
            r"coronary attack",
            r"(coronary|coronary artery) (occlusion|rupture)",
            r"\bMI\b",
            r"\bAMI\b",
            r"(infarction|infarction,|infarction;|infarct|infarctions,|infarcts|infarctions|infarct,|infarcts,) (myocardial|heart)",
            r"(infarction|infarct|rupture) of (heart|the heart|myocardium|the myocardium)",
            r"(cardiac|heart|myocardial) failure",
            r"failure heart",
            r"myocarditis",
            r"pericarditis",
            r"\bHF\b",
            r"\bCHF\b",
            r"ischemic heart disease",
            r"myocardial ischemia",
            r"\bIHD\b",
            r"coronary syndrome",
            r"\bACS\b",
            r"stroke",
            r"cerebrovascular accident",
            r"cerebral infarction",
            r"\bCVA\b"]

'''
Extract any sentences containing at least one defined keyword, and store in a new column.
'''

relevant_sentences = []
for sentences in df['sentence_tokens']:
    current_sentences = []
    
    #Iterate through all sentences of each report
    for sentence in sentences:
        flag = 0
        
        #Iterate through all keywords
        for keyword in keywords:
            match = re.search(keyword, sentence, re.I)

            #Determine if a keyword is present
            if match != None:
                flag = 1
                break;
        if flag == 1:
            current_sentences.append(sentence)
    relevant_sentences.append(current_sentences)

df['relevant_sentences'] = relevant_sentences
