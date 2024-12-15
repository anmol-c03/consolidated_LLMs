import regex as re
from unicodedata import category
from Language_models.tokenization.basic_tokenizer.basic_tokenization import merge

gpt2pat = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def get_stats(ids,counts=None):
    counts={} if counts is None else counts
    for pair in zip(ids,ids[1:]):
        counts[pair]=counts.get(pair,0)+1
    return counts




def make_tokens(compiled_pattern,text):    #return tokenized chunks of text_chunks defined by re.findall()
      text_chunk=re.findall(compiled_pattern, text)
      tokens=[list(chunk.encode('utf-8')) for chunk in text_chunk]
      return tokens

def render(tokens):
    t=tokens.decode('utf-8',errors='replace')
    out=[]
    for token in t:
        if category(token)[0]!='C':
            out.append(token)
        else:
            out.append(f'\\u{ord(token):04x}')
    return ''.join(out)


class special_tokenizer:
    def __init__(self,pattern=None) -> None:
        self.pattern=gpt2pat if pattern is  None else pattern
        self.compiled_pattern=re.compile(self.pattern)
        self.special_tokens={}
        self.inverse_special_tokens={}
        self.vocab={}
        self.merges={}

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def train(self,text,vocab_size) :
        assert vocab_size>=256
        num_iters=vocab_size-256
        tokens=make_tokens(self.compiled_pattern,text)
        ids=list(tokens)
        merges = {} # (int, int) -> int
        vocab={k:bytes([k]) for k in range(256)}
        for i in range(num_iters):
            stats = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            merges[pair] = idx
            vocab[idx]=vocab[pair[0]]+vocab[pair[1]]
        self.merges=merges
        self.vocab=vocab

    def register_special_tokens(self,s):
        self.special_tokens=s
        self.inverse_special_tokens={v:k for k,v in self.special_tokens.items()}

    def decode(self,tokens):
        bytes=[]
        for token in tokens:
            if token in self.vocab:
                bytes.append(self.vocab[token])
            elif token in self.inverse_special_tokens:
                bytes.append(self.inverse_special_tokens[token].encode('utf-8'))
            else:
                raise ValueError('invalid token')
        text_bytes=b''.join(bytes)
        raw_text=text_bytes.decode('utf-8',errors='replace')
        return raw_text
    
    def encode_chunk(self,chunk_tokens):
        ids=list(chunk_tokens)
        while len(ids)>=2:
            stats=get_stats(ids)
            pair=min(stats,key= lambda p: self.merges.get(p,float('inf')))
            if pair not in self.merges:
                break
            target=self.merges[pair]
            ids=merge(ids,pair,target)
        
        return ids

    def encode_ordinary_text(self,text):
        tokens=make_tokens(self.compiled_pattern,text)
        ids=[]
        for chunk_token in tokens:
            ids.extend(self.encode_chunk(chunk_token))
        return ids
        
    def encode(self,text,allowed_special=None):
        special=None
        if allowed_special=='all':
            special=self.special_tokens
        elif allowed_special==None:
            special={}
        else:
            raise ValueError(f'allowed special type  {allowed_special} : Not understood')
        
        if not special:
            return self.encode_ordinary_text(text)
        
        special_pattern='('+'|'.join(re.escape(k) for k in special)+')'

        ## suppose i am using <|end|> as my special token then it will o/p <\\|end\\|>
        # The result is due to re.escape that escapes(puts backslash) characters like '|','.','$', and so on
        # furtermore it is essential because these patterns are using for splitting special tokens from text
        #the use of parenthesis is that re.split that we will use in next loc return the pattern too when pattern
        #is defined inside parenthesis
        ids=[]
        chunks=re.split(special_pattern,text)
        for chunk in chunks:
            if chunk in special:
                ids.append(special[chunk])
            else:
                ids.extend(self.encode_ordinary_text(chunk))
        return ids

    def save(self,filename):
        model_file=filename+'.model'
        with open(model_file,'w',encoding='utf-8') as f:
                f.write('special tokens\n')
                f.write(f'{self.compiled_pattern}\n')
                f.write(f'{len(self.special_tokens)}\n')
                for token in self.special_tokens:
                    f.write(f'{token} {self.special_tokens[token]}\n')
                f.write('merged indices\n')
                for index in self.merges:
                    f.write(f'{index[0]} {index[1]}\n')
                    
        vocab_file=filename+'.vocab'
        invert_merges={v:k for k,v in self.merges.items()}
        with open(vocab_file,'w',encoding='utf-8') as f:
            for index,token in self.vocab.items():
                ch=render(token)
                if index in invert_merges:
                    p0,p1=invert_merges[index]
                    ch0=render(self.vocab[p0])
                    ch1=render(self.vocab[p1])
                    f.write(f'{ch0}+{ch1}->{ch}\n')
                else:
                    f.write(f'{ch}\n')
                    
    def load(self,file_name):
        model_file=file_name+'.model'
        special,merges={},{}
        idx=256
        assert model_file=='special.model'
        with open(model_file,'r') as f:
            type=f.readline().strip()
            assert type=='special tokens'
            pattern=f.readline().strip()
            num_iters=f.readline().strip()
            for i in range(int(num_iters)):
                token,id=f.readline().strip().split(' ')
                special[token]=int(id)
            type=f.readline().strip()
            assert type=='merged indices'
            for indices in f:
                idx0,idx1=map(int,indices.split())
                merges[idx0,idx1]=idx
                idx+=1
        self.merges=merges
        self.pattern=pattern
        self.register_special_tokens(special)
        self.vocab=self._build_vocab()
                
            