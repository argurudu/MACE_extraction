import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

#Load Mistral-7B-Instruct Model
mistral_id = "mistralai/Mistral-7B-Instruct-v0.3"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16)

mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_id)
mistral_model = AutoModelForCausalLM.from_pretrained(
    mistral_id,
    quantization_config=bnb_config,
    device_map="auto")

mistral_model.config.use_cache = False
mistral_model.eval()

#Few-Shot Prompt
def build_mistral_messages(note):
    system_instr = "You are a clinical data extractor. Return ONLY the digit 1, 0, or -1."
    user_content = f"""

### MACE Includes: 
- myocardial infarction (MI / heart attack)
- acute coronary syndrome (ACS)
- acute ischemic or hemorrhagic stroke
- heart failure requiring urgent or emergent care

### MACE Label Rules:
1 = Active/Acute MACE (current encounter)
0 = No Acute MACE (only history, risk, or absent condition)
-1 = Insufficient Info

### Examples:

Text: he was brought to stroke service after presenting with acute deficits.
Label: 1

Text: no chf symptoms; father had MI; mother had stroke.
Label: 0

Text: symptoms of heart attack, including severe chest pain, but further evidence is needed.
Label: -1

Text: female with acute on chronic systolic heart failure admitted to the icu.
Label: 1

Text: history of mi, stroke, heart disease, currently stable.
Label: 0

### Task:
Text: 
{note}

Label:"""

    messages = [
        {"role": "system", "content": system_instr},
        {"role": "user", "content": user_content}
    ]
    return messages


#Generate MACE classification from prompt
note = "[CLINICAL NOTE TEXT]"
message = build_mistral_messages(note)

prompt = mistral_tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False)

inputs = mistral_tokenizer(
    prompt,
    truncation=True,
    max_length=2048,
    return_tensors="pt").to(mistral_model.device)

outputs = mistral_model.generate(
    **inputs,
    max_new_tokens=15,
    pad_token_id=mistral_tokenizer.eos_token_id,
    do_sample=False,
    temperature=0.0,
    use_cache=False)

response = mistral_tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True).strip()

#Process Response into ternary classification
match = re.search(r'-?1|0', response)
if match:
    classification = int(match.group())
else:
    classification = -1

print(response)
print(classification)
