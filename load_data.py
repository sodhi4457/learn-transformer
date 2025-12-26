import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizers import Tokenizer
from datasets import load_from_disk
from torch.utils.data import random_split

max_len = 20
src = 'en'
tgt = 'it'
split_size = 0.9
batch_size = 20

class CustomTokenizer:
    def __init__(self,lang):
        self.tokenizer = Tokenizer.from_file(f"data/tokenizer/tokenizer-{lang}.json")

    def encode(self,text):
        return self.tokenizer.encode(text).ids
    
    def decode(self,ids):
        return self.tokenizer.decode(ids)

class BilingualDataset(Dataset):
    def __init__(self,data):
        self.data = data
        self.src_tokenizer = CustomTokenizer(src)
        self.tgt_tokenizer = CustomTokenizer(tgt)  

        self.sos = torch.tensor([self.src_tokenizer.tokenizer.token_to_id('[SOS]')],dtype=torch.int64)
        self.eos = torch.tensor([self.src_tokenizer.tokenizer.token_to_id('[EOS]')],dtype=torch.int64)
        self.pad = torch.tensor([self.src_tokenizer.tokenizer.token_to_id('[PAD]')],dtype=torch.int64)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        text = self.data[idx]
        src_text = text['translation'][src]
        tgt_text = text['translation'][tgt]

        encoder_input_token = torch.tensor(self.src_tokenizer.encode(src_text),dtype=torch.int64)
        decoder_input_token = torch.tensor(self.tgt_tokenizer.encode(tgt_text),dtype=torch.int64)

        enc_pad_tokens = max_len - len(encoder_input_token)-2
        dec_pad_tokens = max_len - len(decoder_input_token)-1
        
        if enc_pad_tokens < 0 :
            encoder_input_token = encoder_input_token[:max_len-2]
            enc_pad_tokens = 0
            
        if dec_pad_tokens <0:
            decoder_input_token = decoder_input_token[:max_len-1] 
            dec_pad_tokens = 0

        encoder_input = torch.cat(
                            [self.sos,
                            encoder_input_token,
                            self.eos,
                            torch.tensor([self.pad]*enc_pad_tokens,dtype=torch.int64)]
                            , dim=0
                        )
        decoder_input = torch.cat(
                            [self.sos,
                            decoder_input_token,
                            torch.tensor([self.pad]*dec_pad_tokens,dtype = torch.int64)
                            ]
                            , dim=0
                        )
        label = torch.cat(
            [
                decoder_input_token,
                self.eos,
               torch.tensor([self.pad]*dec_pad_tokens,dtype = torch.int64)
            ],
            dim=0,
        )
        # print(f'encoder_input size: {encoder_input.shape}')
        # print(f'encoder_input: {encoder_input}')
        # print(f'decoder_input size: {decoder_input.shape}')
        # print(f'decoder_input: {decoder_input}')
        return {
                'encoder_input':encoder_input,
                'decoder_input':decoder_input,
                'src_text':src_text,
                'tgt_text':tgt_text,
                'label':label,
                'encoder_mask': (encoder_input != self.pad).unsqueeze(0).unsqueeze(0).int(),
                'decoder_mask': (decoder_input != self.pad).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
                

        }
        
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

def get_data_loaders():
    data = load_from_disk('data/en-it')
    size = len(data)
    print('Total Size:',size)
    data =  BilingualDataset(data)
    train_dataset,test_dataset = random_split(data,(split_size,1-split_size))
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size)

    return train_dataloader,test_dataloader

if __name__ == '__main__':
    get_data_loaders()