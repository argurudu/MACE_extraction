#Install if needed: 
!pip install pyConTextNLP

import pyConTextNLP.pyConText as pyConText
import pyConTextNLP.itemData as itemData
import networkx as nx

'''
Define the list of modifiers and targets for pyConText.
'''

modifiers = itemData.get_items(
    "file:///mnt/storage/MACE_extraction/pyConText_itemData/MACE_modifiers.txt")
targets = itemData.get_items(
    "file:///mnt/storage/MACE_extraction/pyConText_itemData/MACE_targets.txt")


'''
Define the pyConText markup_sentence function.
'''

def markup_sentence(s, modifiers, targets, prune_inactive=True):
    markup = pyConText.ConTextMarkup()
    markup.setRawText(s)
    markup.cleanText()
    markup.markItems(modifiers, mode="modifier")
    markup.markItems(targets, mode="target")
    markup.pruneMarks()
    markup.dropMarks('Exclusion')
    markup.applyModifiers()
    markup.pruneSelfModifyingRelationships()
    if prune_inactive:
        markup.dropInactiveModifiers()
    return markup


'''
Exclude sentences whose MACE keyword(s) is modified by definite or probable negated existence. Store only positive sentences in a new column.
'''

list_of_values = []

for sentences_per_report in df['relevant_sentences']:
    list_to_add = []

    #Iterate through every relevant sentence in each report
    for sent in sentences_per_report:
        
        #If a keyword is not associated with any modifier, then it is included as a positive sentence.
        if len(markup_sentence(sent.lower(), modifiers, targets).edges()) < 1:
            list_to_add.append(sent)

        else:
            is_positive = True

            #Iterate through every edge in the sentence
            for edge in (markup_sentence(sent.lower(), modifiers, targets).edges()):

                #If the keyword is accompanied with a negated existence modifier, it is excluded.
                if str(edge[0]).__contains__('definite_negated_existence') or str(edge[0]).__contains__('probable_negated_existence'):
                    is_positive = False
                    break
                  
            if is_positive:
                list_to_add.append(sent)
    list_of_values.append(list_to_add)

df['only_positive'] = list_of_values
  
