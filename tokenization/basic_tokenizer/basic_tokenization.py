from unicodedata import category

def get_stats(ids):
    counts={}
    for pair in zip(ids,ids[1:]):
        counts[pair]=counts.get(pair,0)+1
    return counts

def merge(tokens,pair,target):
    newtokens=[]
    i=0
    while i < len(tokens):
        if i<len(tokens)-1 and tokens[i]==pair[0] and tokens[i+1]==pair[1]:
            newtokens.append(target)
            i+=2
        else:
            newtokens.append(tokens[i])
            i+=1
    return newtokens    

def replace_control_character(s):
    out=[]
    for ch in s:
        if category(ch)[0]!='C':
            out.append(ch)
        else:
            out.append(f'\\u{ord(ch):04x}')
    return ''.join(out)

def render_token(t):
    s=t.decode('utf-8',errors='replace')
    s=replace_control_character(s)
    return t


class basic_tokenizer:
    def __init__(self):
        self.merges={}
        self.vocab={}
    
    def train(self,text,vocab_size):
        assert vocab_size>=256
        num_iters=vocab_size-256
        tokens=list(text.encode('utf-8'))
        target=255
        ids=list(tokens)
        merges={}
        vocab={index:bytes([index]) for index in range(256)}
        for i in range(num_iters):
            if len(ids)<2:
                break
            stats=get_stats(ids)
            pair=max(stats,key=stats.get)
            target=target+1
            ids=merge(ids,pair,target)
            merges[pair]=target
            vocab[target]=vocab[pair[0]]+vocab[pair[1]]
        self.merges=merges
        self.vocab=vocab
        
    def encode(self,text):
        tokens=list(text.encode('utf-8'))
        ids=list(tokens)
        while len(ids)>=2:
            stats=get_stats(ids)
            pair=min(stats,key = lambda p:self.merges.get(p,float('inf')))
            if pair not in self.merges.keys():
                break
            target=self.merges[pair]
            ids=merge(ids,pair,target)
    
        return ids
    
    def decode(self,t):
        index_byte=b''.join(self.vocab[index] for index in t)
        text=index_byte.decode('utf-8',errors='replace')
        return text
        
    def register_special_tokens(self,s):
        pass
    
    def save(self,file_name):
        model_file=file_name+'.model'
        with open(model_file,'w') as f:
            f.write('Binary Pair Encodeing')
            for idx1,idx2 in self.merges.keys():
                f.write(f'{idx1,idx2}\n')

        vocab_file=file_name+'.vocab'
        invert_merges={tokens:pair for pair,tokens in self.merges.items()}
        with open(vocab_file,'w') as f:
            for index,token in self.vocab.items():
                chrs=render_token(token)
                if index in invert_merges:
                    idx0,idx1=invert_merges[index]
                    chr0=render_token(self.vocab[idx0])
                    chr1=render_token(self.vocab[idx1])
                    f.write(f'{chr0} {chr1} -> {chrs}\n')
                else:
                    f.write(f'{chrs}\n')