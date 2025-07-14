import subprocess

list_of_files = ['sentence_tokenization.py', 'relevant_sentence_extraction.py', 'excluding_negation.py', 'MACE_classification.py']
for file in list_of_files:
  subprocess.run(['python', file])
