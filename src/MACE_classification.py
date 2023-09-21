from excluding_negation import df

'''
Classify all positive sentences into 4 categories: historical MACE occurence, current MACE occurence, uncertain MACE occurence, and future risk of MACE occurence. Add each type of sentence to its corresponding column.
'''

historical = []
definite_existence = []
uncertain_existence = []
risk = []
family = []

for sentences_per_report in df['only_positive']:
    historical_to_add = []
    definite_existence_to_add = []
    uncertain_existence_to_add = []
    risk_to_add = []
    family_to_add = []
    
    #Iterate through all relevant sentences
    for sent in sentences_per_report:
        
        #Classify as a definite existence if the keyword is not accompanied with a modifier
        if len(markup_sentence(sent.lower(), modifiers, targets).edges()) < 1:
            definite_existence_to_add.append(sent)
        else:
            
            #Iterate through every edge
            for edge in (markup_sentence(sent.lower(), modifiers, targets).edges()):
                used = False

                #Classify as historical if historical modifier is present
                if str(edge[0]).__contains__('historical'):
                    historical_to_add.append(sent)
                    used = True
                    break
                
                #Classify as family history if family history modifier is present
                elif str(edge[0]).__contains__('family_history'):
                    family_to_add.append(sent)
                    used = True
                    break
                    
                #Classify as future risk if future modifier is present
                elif str(edge[0]).__contains__('future'):
                    risk_to_add.append(sent)
                    used = True
                    break
                
                #Classify as uncertain existence if uncertainty modifiers are present
                elif str(edge[0]).__contains__('indication') or str(edge[0]).__contains__('ambivalent_existence'):
                    uncertain_existence_to_add.append(sent)
                    used = True
                    break
            
            if not used:
                definite_existence_to_add.append(sent)
   
    historical.append(historical_to_add)
    definite_existence.append(definite_existence_to_add)
    uncertain_existence.append(uncertain_existence_to_add)
    risk.append(risk_to_add)
    family.append(family_to_add)

df['MACE_historical'] = historical
df['MACE_current_existence'] = definite_existence
df['MACE_uncertain_existence'] = uncertain_existence
df['MACE_risk'] = risk
df['MACE_family_history'] = family


'''
Provide ternary classification for each report. 
1: Positive MACE finding, only if there is a current existence
0: Negative MACE finding, includes reports with no MACE terms or historical/future occurences of MACE
-1: Uncertain MACE finding, only if there are uncertain modifiers 
'''

t_classification = []

#Iterate through every report
for i in range(len(df)):

    #Classify as positive if the current_existence column contains sentences
    if len(df['MACE_current_existence'][i]) > 0:
        t_classification.append(1)
    
    #Classify as uncertain if the uncertain_existence column contains sentences
    elif len(df['MACE_uncertain_existence'][i]) > 0:
        t_classification.append(-1)
    
    #Otherwise classify as negative
    else:
        t_classification.append(0)

df['validation'] = t_classification
