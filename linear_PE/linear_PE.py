import torch
import matplotlib.pyplot as plt

def linear_pos_enc(context_window,channels):#(T,C) on accordance to gpt2 channels are often called as d_model
    PE=torch.zeros(context_window,channels)
    for pos in range(context_window):
        for i in range(channels//2):
            angle=torch.tensor(pos/1000**(2*i/channels))
            '''
                why use sin and cos ??
                    common question arises.
                    1.This is due to the fact that sin and cos are orthogonal to each other hence no two
                    consecutive dimension have same have value in that embedding space
                    2.another fact is both are periodic hence lower dimensions will results in repeating 
                    dimensional embeding after certain token 
                            i.e for i->0(zeroth dimension)
                            sin(angle)=sin(pos)
                            since the  freq is high for zeroth dimension it highly oscillates and will 
                            result in repeating values after certain no. of tokens or we can say after 
                            its time peroid capturing finer positional details

                            for higher dimensional space 
                            lets take i=12
                            1000^(2*12/channels) 
                            such that denominator will have high value results in very small angle that 
                            makes frequency low 
                            and for very small angle
                            if x->0
                            sin(x) equivalent to x
                            hence it is linear function with non repeating values
                            so what does it tells us??
                            it tells that broader positional details are enoded by higher dimensions
                            which will be clear after looking in visualization in its .ipynb file

                    3.The most important reason behind using sin and cosine is 
                        PE(pos+k) is represented as linear function of PE(pos) 
                        PE(pos+k,2i)=sin((pos+k)/(1000^(2i/dmodel)))

                        we know
                            sin(a+b)=sin(a)cos(b)+sin(b)cos(a)
                            let a=pos/(1000^(2i/dmodel))
                            let b=k/(1000^(2i/dmodel))
                        
                            such that 
                            PE(pos+k) can be represented as linear function of PE(pos)
            '''
            PE[pos,2*i]=torch.sin(angle)
            PE[pos,2*i+1]=torch.cos(angle)
    
    return PE

context_window=8
channels=32

pos_enc=linear_pos_enc(context_window,channels)

# visaulization of positional  embedding
even_pos_enc=pos_enc[:,::2]
odd_pos_enc=pos_enc[:,1::2]

plt.figure(figsize=(15, 7))
for i in range(even_pos_enc
.shape[1]):
    plt.plot(range(context_window), even_pos_enc[:, i], label=f'Dimension {2*i}')
plt.title('Positional Encoding - Even Positions')
plt.xlabel('Position')
plt.ylabel('Encoding Value')
# plt.legend(loc='upper right')
# plt.savefig('even_pos_enc')
# plt.show()

# Plot odd positions
plt.figure(figsize=(15, 7))
for i in range(odd_pos_enc.shape[1]):
    plt.plot(range(context_window), odd_pos_enc[:, i], label=f'Dimension {2*i+1}')
plt.title('Positional Encoding - Odd Positions')
plt.xlabel('Position')
plt.ylabel('Encoding Value')
# plt.legend(loc='upper right')
# plt.savefig('odd_pos_enc')
# plt.show()

plt.plot(pos_enc);
# plt.savefig('linear_pos_enc')
# plt.show()

# To see the plot one can either visit .ipynb or uncomment plt.savefig() and plt.show()