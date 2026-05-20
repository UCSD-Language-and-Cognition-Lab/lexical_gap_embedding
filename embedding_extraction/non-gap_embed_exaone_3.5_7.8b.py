############################################################################################################################################################
# Create a dictionary (.pt file) of contextualized embeddings for the non-gap words (adjectives) that pass both 'I feel' and 'I am' acceptability judgments.
# The dictionary will contain the last hidden states of the last layer, first layer, and the average of the last 4 layers for each word.
############################################################################################################################################################

import pandas as pd
import re
import os
import glob
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.tokenize import word_tokenize
import math 

##### load the word data #####
# download the dataset from https://github.com/yoonwonj/EVOKE

# process the word data (this data contains translational mappings and POS information)
data = pd.read_excel('./Word_dataset/Emotion_Words_Combined_Clean.xlsx') #replace with your directory
data['Korean_Word'] = data['Korean_Word'].apply(lambda w: re.sub(r'\*', '', str(w)))
data['English_Word'] = data['English_Word'].apply(lambda w: re.sub(r'\*', '', str(w)))

# filter for non-gap words, adjectives only
equivdf= data[data.Translational_Equivalence==1]
equiadj=equivdf[equivdf.POS=='adjective']

# annotation dataset (adjectives only)
edata = pd.read_csv('./Word_dataset/English_annotation/adj-English-final-combined.csv', index_col=False) #replace with your directory
kdata= pd.read_csv('./Word_dataset/Korean_annotation/adj-Korean-final-combined.csv', index_col=False) #replace with your directory

# replace 'not sure' with 'unsure'
kdata.replace('not sure', 'unsure', inplace=True)
edata.replace('not sure', 'unsure', inplace=True)

# preprocess the words and filter for adjectives only
kdata['Korean_Word'] = kdata['Korean_Word'].apply(lambda w: re.sub(r'\*', '', str(w)).lower())
edata['Word'] = edata['Word'].apply(lambda w: re.sub(r'\*', '', str(w)).lower())
kadj=kdata[kdata.POS=='adjective']
eadj=edata[edata.POS=='adjective']

##### load the model #####
# Run download_model_exaone_3.5_7.8b.py first to download the model and save it in the cache directory.
# Then run this code below to load the model from the cache directory.

model_name = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"

# locate the cached directory
base_cache = os.path.expanduser("~/.cache/huggingface/hub")
pattern = os.path.join(
    base_cache,
    f"models--{model_name.replace('/', '--')}",
    "snapshots",
    "*"
)
matches = glob.glob(pattern)

if len(matches) == 0:
    raise ValueError(f"Model not found in cache: {pattern}")

model_dir = matches[0]
print("Using local model directory:", model_dir)

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir) #replace 'model_dir' with 'model_name' if 'model_dir' throws an error

##### Filter the words based on the annotations on acceptability judgments: 'I feel' and 'I am' #####
### filter out words that are not acceptable in 'I feel' context ###
ifeel_e= eadj[eadj['I_Feel']=='acceptable']
ifeel_k= kadj[kadj['I_Feel']=='acceptable']

# preprocess the words 
kor_adj= [word for word in ifeel_k.Korean_Word.values.astype(str).tolist()]
eng_adj= [word.lower() for word in ifeel_e.Word.values.astype(str).tolist()]

kor_adj= set(kor_adj)
eng_adj= set(eng_adj)

# Exclude NaN values
kor_adj = [w for w in kor_adj if not (w is None or (isinstance(w, float) and math.isnan(w)))]
eng_adj = [w for w in eng_adj if not (w is None or (isinstance(w, float) and math.isnan(w)))]

kor_temp= list(ifeel_k.Korean_Word)
eng_temp= list(ifeel_e.Word)

# create boolean filters to only include words that have passed 'I feel' tests (based on the annotations from the dataset)
bool_mask_list_k = []

for word in kor_temp:
    equi_ewordlist = equiadj[equiadj['Korean_Word'] == word]['English_Word'].values
    if ifeel_e['Word'].isin(equi_ewordlist).any():
        bool_mask_list_k.append(True)
    else:
        bool_mask_list_k.append(False)

bool_mask_list_e = []

for word in eng_temp:
    equi_kwordlist = equiadj[equiadj['English_Word'] == word]['Korean_Word'].values
    if ifeel_k['Korean_Word'].isin(equi_kwordlist).any():
        bool_mask_list_e.append(True)
    else:
        bool_mask_list_e.append(False)

# filter the words based on the boolean filters
ifeel_e=ifeel_e[bool_mask_list_e]
ifeel_k=ifeel_k[bool_mask_list_k]
print(f"'I feel' - {len(ifeel_e)} English words, {len(ifeel_k)} Korean words")

### filter out words that are not acceptable in 'I am' context ###
iam_e= ifeel_e[ifeel_e['I_Am']=='acceptable']
iam_k= ifeel_k[ifeel_k['I_Am']=='acceptable']

# preprocess the words 
kor_adj= [word for word in iam_k.Korean_Word.values.astype(str).tolist()]
eng_adj= [word.lower() for word in iam_e.Word.values.astype(str).tolist()]

kor_adj= set(kor_adj)
eng_adj= set(eng_adj)

# Exclude NaN values
kor_adj = [w for w in kor_adj if not (w is None or (isinstance(w, float) and math.isnan(w)))]
eng_adj = [w for w in eng_adj if not (w is None or (isinstance(w, float) and math.isnan(w)))]

kor_temp= list(iam_k.Korean_Word)
eng_temp= list(iam_e.Word)

# create boolean filters to only include words that have passed 'I feel' tests (based on the annotations from the dataset)
bool_mask_list_k = []

for word in kor_temp:
    equi_ewordlist = equiadj[equiadj['Korean_Word'] == word]['English_Word'].values
    if iam_e['Word'].isin(equi_ewordlist).any():
        bool_mask_list_k.append(True)
    else:
        bool_mask_list_k.append(False)

bool_mask_list_e = []

for word in eng_temp:
    equi_kwordlist = equiadj[equiadj['English_Word'] == word]['Korean_Word'].values
    if iam_k['Korean_Word'].isin(equi_kwordlist).any():
        bool_mask_list_e.append(True)
    else:
        bool_mask_list_e.append(False)
iam_e=iam_e[bool_mask_list_e]
iam_k=iam_k[bool_mask_list_k]

print(f"both 'I feel' and 'I am' - {len(iam_e)} English words, {len(iam_k)} Korean words")

##### Get the contextualized embeddings of the filtered words #####
##### Only use words that pass both 'I am' and 'I feel' test #####
##### Use the same sentence template ("I am [word]." and "나는 [word].") for both Korean and English to extract the contextualized embeddings #####
iam_korean_list=list(set(iam_k.Korean_Word))
iam_english_list=list(set(iam_e.Word))

# dictionary of embeddings
iam_embed_dict={}

### Korean ### 
for i, word in enumerate(iam_korean_list):
    # Skip if the word already exists in the dictionary
    if word in iam_embed_dict:
        continue
    
    # Get the correct wordform for the Korean word to fit the sentence template
    wordform= iam_k[iam_k.Korean_Word==word]['I_Am_Wordform'].values[0].strip()
    # if the wordform is NaN, raise an error
    if isinstance(wordform, float) and math.isnan(wordform):
        raise ValueError(f"Wordform for '{word}' is NaN. Please check the dataset.")
    
    # create the sentence with the correct wordform
    kor_sentence= "나는 "+wordform+"."
    # prepare the input for the model
    inputs = tokenizer(kor_sentence, return_tensors='pt').to("cuda")

    # extract embeddings
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # last hidden states of the last layer, first layer, and the average of the last 4 layers
    last_hidden_states_kor = outputs.hidden_states[-1]
    first_layer_kor = outputs.hidden_states[0] 
    last_4 = outputs.hidden_states[-4:]
    avg_hidden_states_kor = torch.stack(last_4).mean(0)

    # get the embeddings of the word (slicing to only include the tokens for the word - every model will have different slicing based on the tokenization of the sentence template)
    # change the slicing by the model
    embedding_kor = last_hidden_states_kor[:,2:-1,:].cpu()  ## revise the slicing
    embed_kor_first = first_layer_kor[:,2:-1,:].cpu() ## revise the slicing
    embedding_kor_last4 = embedding_kor_last4 = avg_hidden_states_kor[:,2:-1,:].cpu() ## revise the slicing

    #append the result as dictionary
    iam_embed_dict[word]= {'language':'Korean', 'embed': embedding_kor, 'embed_first':embed_kor_first, 'embed_last4': embedding_kor_last4}

### English ###
for i, word in enumerate(iam_english_list):
    # Skip if the word already exists in the dictionary
    if word in iam_embed_dict:
        continue
        
    # create the sentence with the word
    eng_sentence="I am "+word+"."
    # prepare the input for the model
    inputs = tokenizer(eng_sentence, return_tensors='pt').to("cuda")

    # extract embeddings
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
    # last hidden states of the last layer, first layer, and the average of the last 4 layers
    last_hidden_states_eng = outputs.hidden_states[-1]
    first_layer_eng = outputs.hidden_states[0]
    last_4 = outputs.hidden_states[-4:]     
    avg_hidden_states_eng = torch.stack(last_4).mean(0) 

    # get the embeddings of the word (slicing to only include the tokens for the word - every model will have different slicing based on the tokenization of the sentence template)
    # change the slicing by the model
    embedding_eng = last_hidden_states_eng[:,2:-1,:].cpu() ## revise the slicing
    embed_eng_first= first_layer_eng[:,2:-1,:].cpu() ## revise the slicing
    embedding_eng_last4= avg_hidden_states_eng[:,2:-1,:].cpu() ## revise the slicing

    #append the result as dictionary
    iam_embed_dict[word]= {'language':'English', 'embed': embedding_eng, 'embed_first': embed_eng_first, 'embed_last4': embedding_eng_last4}

# save the dictionary of embeddings
torch.save(iam_embed_dict, 'iam_embed_adj_exaone_3.5_7.8b.pt') 

###################################################################################################################################################################################################################
# Combine the Korean and English non-gap word embeddings (saved as a result of running the code above) into a new paired dictionary by its translations in the other language.
# The gap embeddings (that will be saved as a result of running gap_iam_embed_exaone_3.5_7.8b.py file) do not need to go through this process, as gap words do not have translations in the other language.
###################################################################################################################################################################################################################

# lists of Korean and English words: iam_korean_list and iam_english_list from the code above will be used

# Create a new dictionary to store the paired embeddings
paired_embed_dict = {}

##### pair the embeddings #####
# note that the pair is stored bidirectionally (i.e., under both the Korean and English word entries) for easier access in the future analyses.

### Korean → English direction ###
# find all English translations of the given Korean word and add the English embeddings to the dictionary under the Korean word entry
for i, word in enumerate(iam_korean_list):
    if word not in iam_embed_dict:
        continue

    eng_matches = equiadj.loc[equiadj['Korean_Word'] == word, 'English_Word'].dropna().unique()
    if len(eng_matches) == 0:
        continue

    kor_vector = iam_embed_dict[word]['embed']
    kor_vector_last4 = iam_embed_dict[word]['embed_last4']
    kor_vector_first = iam_embed_dict[word]['embed_first']

    # ensure Korean entry exists
    if word not in paired_embed_dict:
        paired_embed_dict[word] = {'language': 'Korean', 
                                   'kor_vector': kor_vector, 'kor_vector_last4': kor_vector_last4, 'kor_vector_first': kor_vector_first,
                                   'eng_words': {}}

    for eng_word in eng_matches:
        if eng_word not in iam_embed_dict:
            continue
        eng_vector = iam_embed_dict[eng_word]['embed']
        eng_vector_last4 = iam_embed_dict[eng_word]['embed_last4']
        eng_vector_first = iam_embed_dict[eng_word]['embed_first']

        # update embedded dictionary
        paired_embed_dict[word]['eng_words'][eng_word] = {
            'eng_vector': eng_vector,
            'eng_vector_last4': eng_vector_last4,
            'eng_vector_first': eng_vector_first
        }


### English → Korean direction ###
# find all Korean translations of the given English word and add the Korean embeddings to the dictionary under the English word entry
for i, word in enumerate(iam_english_list):
    if word not in iam_embed_dict:
        continue

    kor_matches = equiadj.loc[equiadj['English_Word'] == word, 'Korean_Word'].dropna().unique()
    if len(kor_matches) == 0:
        continue

    eng_vector = iam_embed_dict[word]['embed']
    eng_vector_last4 = iam_embed_dict[word]['embed_last4']
    eng_vector_first = iam_embed_dict[word]['embed_first']

    # ensure English entry exists
    if word not in paired_embed_dict:
        paired_embed_dict[word] = {'language':'English',
                                   'eng_vector': eng_vector, 'eng_vector_last4': eng_vector_last4, 'eng_vector_first': eng_vector_first,
                                   'kor_words': {}}

    for kor_word in kor_matches:
        if kor_word not in iam_embed_dict:
            continue
        kor_vector = iam_embed_dict[kor_word]['embed']
        kor_vector_last4 = iam_embed_dict[kor_word]['embed_last4']
        kor_vector_first = iam_embed_dict[kor_word]['embed_first']

        # update embedded dictionary
        paired_embed_dict[word]['kor_words'][kor_word] = {
            'kor_vector': kor_vector,
            'kor_vector_last4': kor_vector_last4,
            'kor_vector_first': kor_vector_first
        }

# save the paired dictionary of embeddings
torch.save(paired_embed_dict, 'iam_embed_adj_paired_exaone_3.5_7.8b.pt')

# count how many Korean and English entries exist
kor_keys = [k for k in paired_embed_dict.keys() if k in iam_korean_list]
eng_keys = [k for k in paired_embed_dict.keys() if k in iam_english_list]

print(f"Korean entries: {len(kor_keys)}")
print(f"English entries: {len(eng_keys)}")
print(f"Total entries: {len(paired_embed_dict)}")
