## add subscrpiptino key 
import torch
import re

from transformers.models.longformer.tokenization_longformer import LongformerTokenizer
subscription_key = "840577882d7a45a4b07f011f1dc3fc87"
assert subscription_key
search_url = "https://api.bing.microsoft.com/v7.0/search"
from bs4 import BeautifulSoup
import requests

## determine if stuff's elements exist in stuffa
def check_list(sentence, stuffa):
    for element in sentence.split(" "):
        if element in stuffa:
            return False
    return True
## question
questions = [
    "Who killed Jon snow?",
    "How many season did game throne have?"
    "Who is the author of game of throne?",
]

## constants
## remove useless sentences
useless = ["facebook" , "email", "like", "comment", 
"policy", "book", "islamic", "sites","twitter", "fan",
"fbi", ".com" , "science" , "subscription" , "notification",
"blacklist", "instagram" , "icon" , "hulu" , 
"program" , "click", "2021" , "covid" , "twit" , "porn",
 "notification" , "fear the walking dead", "hbo", "tax", "article" , "spoil",
 "advertise", "TV", "follow", "theather" , "tab" , "link", "iframe", "amy" , "skip" , "vox", "Amy" , "Brady",
 "Dog" , "Marvel", "Netflix" , "anime", "Destiny" , "Facebook" , "Twitter" , "anime" , "Vox"
 "Info", "Activation" , "Marvel" ,  "CD" , "Cyberpunk" , "Netflix", "Marvel's" , "Activision",  
 "Editorial" , "Galaxy" , "horror anime" , "Personal", "toxic" , "Myers" , "Snyder"
 "2.0" , "Netflix"
]


import requests 
search_term = "Who killed john snow?"
headers = {"Ocp-Apim-Subscription-Key": subscription_key}
params = {"q": search_term, "textDecorations": True, "textFormat": "HTML"}
response = requests.get(search_url, headers=headers, params=params)
response.raise_for_status()
search_results = response.json()
print(search_results["webPages"]["value"][0]["url"] )

## get the text of a website 
def get_text_of_website(website_url):
    try:
        page = requests.get(website_url)
        soup = BeautifulSoup(page.content, 'html.parser')
        text=  soup.get_text()
        ## preprocess beautiful soap to a single text string
        text_list = text.split("\n")
        ## remove all space size greater than one from a list of string 
        text_list = [re.sub(r'\s+', ' ', x) for x in text_list]
        ## remove all tab of all element of a  list of string
        text_list = [element.replace("\t" , "") for element in text_list]
        ## remove all element of a list of string that have size less than 20
        text_list = [element for element in text_list if len(element)>20]
        ## remove all element that have word from the list of string useless from a list of string
        text_list = [element for element in text_list if check_list(element , useless)]
        return text_list
    except:
        return "Error"
## a method that takes in a list of string and return a single string
def list_to_string(list_of_string):
    return " ".join(list_of_string)





text = get_text_of_website( search_results["webPages"]["value"][0]["url"] )
print( text)
print(len( text ) )
text = list_to_string(text)
print( len ( text ) )
print( text )
###

'''
from transformers import pipeline

question_text = "Who killed john snow?"

question_answer = pipeline("question-answering")

result  = question_answer( question = question_text , context = text )
print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")
'''
from transformers import LongformerTokenizer, LongformerForQuestionAnswering

tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")
model = LongformerForQuestionAnswering.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")

for question in questions:
    encoding = tokenizer(question , text , return_tensors = "pt")
    inputs_ids = encoding["input_ids"]

    ## defaultr is alocal attenion wverywhere
    #the forward method will automatically set global attenion on question token
    attenion_mask = encoding["attention_mask"]
    
    output = model(inputs_ids ,   attenion_mask)
    start_logit = output.start_logit
    end_logit = output.end_logit
    all_token = tokenizer.convert_ids_to_tokens(inputs_ids.tolist()[0])

    answer_text = all_token[torch.argmax(start_logit) : torch.argmax(end_logit)+1]
    answer = tokenizer.decode( tokenizer.convert_tokens_to_ids(answer_text) ) # remove space prepending space token


