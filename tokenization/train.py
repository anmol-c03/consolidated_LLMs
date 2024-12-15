import os
from basic_tokenization import basic_tokenizer
from regex_tokenizer import special_tokenizer

special_tokens = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

text= open('taylorswift.txt','r',encoding='utf-8').read()

for tokenizer_name,name in zip([basic_tokenizer,special_tokenizer],['basic','special']):
    tokenizer=tokenizer_name()
    tokenizer.train(text,512)
    tokenizer.register_special_tokens(special_tokens)
    tokenizer.save(name)

    print(tokenizer.decode(tokenizer.encode('I am anmol chalise. I am currently studying computer engineering')))