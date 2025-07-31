import tiktoken

def tokenize():
    tokenizer  = tiktoken.get_encoding('gpt-2')
    return tokenizer