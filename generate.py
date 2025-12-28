import os
import torch
import torch.nn.functional as F
from transformer import TransformerPipeline
from load_data import CustomTokenizer

def load_checkpoint_for_evaluation(model_instance, checkpoint_path, device):
    """
    Loads a checkpoint for evaluation (inference).
    
    Args:
        model_instance: An initialized instance of your model class.
        checkpoint_path (str): The path to the saved checkpoint file.
        device (torch.device): The device to load the model onto (e.g., 'cuda' or 'cpu').
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # 1. Load the checkpoint dictionary
    # Use map_location to ensure it loads correctly regardless of how it was saved (GPU/CPU)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 2. Load the model parameters
    model_instance.load_state_dict(checkpoint['model_state_dict'])
    
    # 3. Move the model to the specified device
    model_instance.to(device)
    
    # 4. Set the model to evaluation mode (CRITICAL STEP)
    # This disables layers like Dropout and sets BatchNorm to use saved stats.
    model_instance.eval() 
    
    print(f"Model successfully loaded from epoch {checkpoint['epoch']} and set to evaluation mode.")
    return model_instance

if __name__ == '__main__':
    
    # 2. Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model =  TransformerPipeline().to(device)
    # 3. Specify the epoch you want to evaluate (e.g., epoch 500)
    epoch_to_check = 1000
    path = os.path.join('model_v2_checkpoints/', f'epoch_{epoch_to_check:04d}.pth')

    # 4. Load the model
    # model = load_checkpoint_for_evaluation(model, path, device)
    src_tokenizer = CustomTokenizer('en')
    tgt_tokenizer = CustomTokenizer('it')
    sos = torch.tensor([src_tokenizer.tokenizer.token_to_id('[SOS]')],dtype=torch.int64)
    eos = torch.tensor([src_tokenizer.tokenizer.token_to_id('[EOS]')],dtype=torch.int64)
    pad = torch.tensor([src_tokenizer.tokenizer.token_to_id('[PAD]')],dtype=torch.int64)

    
    def text_to_tensor(text):
        src_tokens = torch.tensor(src_tokenizer.encode(text),dtype=torch.long)
        src_len = len(src_tokens)
        pad_len = 20 - src_len -2
        encode_input = torch.cat([sos,src_tokens,eos,torch.tensor([pad]*pad_len,dtype=torch.long)])
        decode_input = torch.empty(1, 1,dtype=torch.long).fill_(tgt_tokenizer.tokenizer.token_to_id('[SOS]'))
        return {
            'encoder_input':encode_input.unsqueeze(0),
            'decoder_input':decode_input,
            'encoder_mask':(encode_input != pad).unsqueeze(0).unsqueeze(0).int(),
            'decoder_mask':(decode_input != pad).unsqueeze(0).unsqueeze(0).int()

        }
        

    x = 'Hence it is that for so long a time, and during so'
    x = text_to_tensor(x)
    enc_input = x['encoder_input'].to(device)
    enc_mask = x['encoder_mask'].to(device)
    dec_input = x['decoder_input'].to(device)
    dec_mask = x['decoder_mask'].to(device)
    encode_output = model.encode(enc_input,enc_mask)
    count = 0
    while len(dec_input[0])<20:
        decode_output = model.decode(dec_input,dec_mask,encode_output,enc_mask)
        probs = F.softmax(decode_output,dim=-1)[0][0]
        idx_next = torch.argmax(probs,dim=-1)
        dec_input = torch.cat([dec_input,idx_next.unsqueeze(0).unsqueeze(0)],dim=1)
    
    print('IT Text:',tgt_tokenizer.decode(dec_input.tolist()[0]))