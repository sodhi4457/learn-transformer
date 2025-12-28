import os
import time
import torch
import torch.nn.functional as F
from transformer import TransformerPipeline
from load_data import get_data_loaders
from load_data import CustomTokenizer
device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'

def load_checkpoint(model_instance, checkpoint_path, device):
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_instance.load_state_dict(checkpoint['model_state_dict'])
    model_instance.to(device)
    print(f"Model successfully loaded from epoch {checkpoint['epoch']}")
    return model_instance

if __name__ == '__main__':
    tokenizer_src = CustomTokenizer('en')
    train_dataloader,test_dataloader = get_data_loaders()

    model = TransformerPipeline().to(device)
    # model = load_checkpoint(model,'model_v2_checkpoints\epoch_0100.pth',device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3)
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=tokenizer_src.tokenizer.token_to_id('[PAD]'))

    model.train()
    start = time.time()
    for epoch in range(1001):
        count = 0
        for batch in train_dataloader:
            # Setting gradint to Zero
            
            optimizer.zero_grad()

            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)
            encoder_output = model.encode(encoder_input,encoder_mask)
            output = model.decode(decoder_input,decoder_mask,encoder_output,encoder_mask,)
            B,T,C = output.shape
            loss = loss_func(output.view(B*T,C),label.view(B*T))
            loss.backward()
            optimizer.step()
            if count > 100:
                break
            count+=1

        if epoch % 100 == 0 and epoch !=0:
            print(f'Epoch {epoch} ;; loss: {loss.item()}')
            checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # Add other data if needed (e.g., best_loss, lr_scheduler_state)
                    }
                            
            # 2. Define the filename with the epoch number
            checkpoint_dir = 'model_v2_checkpoints'
            filename = f'epoch_{epoch:04d}.pth'
            checkpoint_path = os.path.join(checkpoint_dir, filename)

            # 3. Save the checkpoint file
            torch.save(checkpoint, checkpoint_path)
            print(f"-> Checkpoint saved at epoch {epoch} to {checkpoint_path}")
    print(f'time taken:{time.time()-start}, loss: {loss.item()}')