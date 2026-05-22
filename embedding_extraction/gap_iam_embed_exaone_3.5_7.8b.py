########################################################################################################################################################
# Create a dictionary (.pt file) of contextualized embeddings for the gap words (adjectives) that pass both 'I feel' and 'I am' acceptability judgments.
# The dictionary will contain the last hidden states of the last layer, first layer, and the average of the last 4 layers for each word.
########################################################################################################################################################

import pandas as pd
import re
import os
import glob
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

##### load the word data #####
# download the dataset from https://github.com/yoonwonj/EVOKE

# process the word data (this data contains translational mappings and POS information)
data = pd.read_excel('./Word_dataset/Emotion_Words_Combined_Clean.xlsx') #replace with your directory
data['Korean_Word'] = data['Korean_Word'].apply(lambda w: re.sub(r'\*', '', str(w)))
data['English_Word'] = data['English_Word'].apply(lambda w: re.sub(r'\*', '', str(w)))

# filter for non-gap words, adjectives only
gap= data[data.Translational_Equivalence==0]
gapadj=gap[gap.POS=='adjective']

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

# Locate the cached directory
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
    trust_remote_code=True,
    local_files_only=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name) # if model_name doesn't work for your model, try model_dir instead (+ add trust_remote_code=True and local_files_only=True)

##### Preprocess words in the gap dataframe #####
# whether the words are unique only in Korean or in English
# check if a string is exactly one word without spaces, slashes, or hyphens
def is_single_word(x):
    # Exclude NA/NaN/none values and non-string values
    if pd.isna(x):
        return False
    x = str(x).strip()
    if x.lower() in {"nan", "none"}:
        return False
    # Reject if contains space, slash, hyphen, or apostrophe
    if re.search(r"[ \-/']", x):
        return False
    return True
   
# create separate dataframes for Korean and English gap adjectives (no translational equivalent in the other language)
cond_kor = gapadj['Korean_Word'].apply(is_single_word)
gapadjkor= gapadj[cond_kor]

cond_eng = gapadj['English_Word'].apply(is_single_word)
gapadjeng= gapadj[cond_eng]

overlap = gapadj[cond_kor & cond_eng]
if len(overlap) > 0:
    print("Warning: There are words that are single-word in both Korean and English columns. Please check the following entries:")
    print(overlap[['Korean_Word', 'English_Word']])

# preprocess the words 
kor_adj_gap= set(gapadjkor.Korean_Word.values.tolist())
eng_adj_gap= [word.lower() for word in gapadjeng.English_Word.values.astype(str).tolist()] #lowercase
eng_adj_gap= set(eng_adj_gap)

 # Exclude NaN values
kor_adj_gap = [w for w in kor_adj_gap if not (w is None or (isinstance(w, float) and math.isnan(w)))]
eng_adj_gap = [w for w in eng_adj_gap if not (w is None or (isinstance(w, float) and math.isnan(w)))]
print("Number of words without translational equivalent(s) in the other language")
print(f"Korean adjectives: {len(kor_adj_gap)}, English adjectives: {len(eng_adj_gap)}")

##### Filter the words based on the annotations on acceptability judgments: 'I feel' and 'I am' #####
### filter out words that are not acceptable in 'I feel' context ###
ifeel_e= eadj[eadj['I_Feel']=='acceptable']
ifeel_k= kadj[kadj['I_Feel']=='acceptable']

# 'I feel' acceptable list
kor_temp= list(ifeel_k.Korean_Word)
eng_temp= list(ifeel_e.Word)

kor_adj_gap_ifeel=[]

# append the Korean words that are in the 'I feel' acceptable list
for word in kor_adj_gap:
    if word in kor_temp:
        kor_adj_gap_ifeel.append(word)

eng_adj_gap_ifeel=[]

# append the English words that are in the 'I feel' acceptable list
for word in eng_adj_gap:
    if word in eng_temp:
        eng_adj_gap_ifeel.append(word)

print(f"Korean words in 'I feel' acceptable list: {len(kor_adj_gap_ifeel)}, English words in 'I feel' acceptable list: {len(eng_adj_gap_ifeel)}")

### filter out words that are not acceptable in 'I am' context ###
iam_e= eadj[eadj['I_Am']=='acceptable']
iam_k= kadj[kadj['I_Am']=='acceptable']

# 'I am' acceptable list
kor_temp= list(iam_k.Korean_Word)
eng_temp= list(iam_e.Word)

kor_adj_gap_iam=[]

# append the Korean words that are in the 'I am' acceptable list
for word in kor_adj_gap_ifeel:
    if word in kor_temp:
        kor_adj_gap_iam.append(word)

eng_adj_gap_iam=[]

# append the English words that are in the 'I am' acceptable list
for word in eng_adj_gap_ifeel:
    if word in eng_temp:
        eng_adj_gap_iam.append(word)

print(f"Korean words in both 'I feel' and 'I am' acceptable lists: {len(kor_adj_gap_iam)}, English words in both 'I feel' and 'I am' acceptable lists: {len(eng_adj_gap_iam)}")

##### Get the contextualized embeddings of the filtered gap words #####
##### Only use words that pass both 'I am' and 'I feel' test #####
##### Use the same sentence template ("I am [word]." and "나는 [word].") for both Korean and English to extract the contextualized embeddings #####

# final gap word lists: kor_adj_gap_iam, eng_adj_gap_iam

#dictionary of embeddings
iam_embed_dict_gap={}

#Korean
for i, word in enumerate(kor_adj_gap_iam):
    # Skip if the word already exists in the dictionary
    if word in iam_embed_dict_gap:
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
    embed_kor_first = first_layer_kor[:,2:-1,:].cpu()  ## revise the slicing
    embedding_kor_last4 = embedding_kor_last4 = avg_hidden_states_kor[:,2:-1,:].cpu()  ## revise the slicing

    #append the result as dictionary
    iam_embed_dict_gap[word]= {'language':'Korean', 'embed': embedding_kor, 'embed_first':embed_kor_first, 'embed_last4': embedding_kor_last4}


#English
for i, word in enumerate(eng_adj_gap_iam):
    # Skip if the word already exists in the dictionary
    if word in iam_embed_dict_gap:
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
    iam_embed_dict_gap[word]= {'language':'English', 'embed': embedding_eng, 'embed_first': embed_eng_first, 'embed_last4': embedding_eng_last4}

# save the dictionary of embeddings
torch.save(iam_embed_dict_gap, './iam_embed_adj(gap)_exaone_3.5_7.8b_updated.pt') #change this to your own directory, match the model version