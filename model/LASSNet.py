import torch
import torch.nn as nn
import torch.nn.functional as F
from .text_encoder import Text_Encoder
from .resunet_film import UNetRes_FiLM

class LASSNet(nn.Module):
    def __init__(self, device='cuda'):
        super(LASSNet, self).__init__()
        self.text_embedder = Text_Encoder(device)
        self.UNet = UNetRes_FiLM(channels=1, cond_embedding_dim=256)

    def forward(self, x, caption):
        # x: (Batch, 1, T, 128))
        input_ids, attns_mask = self.text_embedder.tokenize(caption)
        
        cond_vec = self.text_embedder(input_ids, attns_mask)[0]
        dec_cond_vec = cond_vec

        mask = self.UNet(x, cond_vec, dec_cond_vec)
        mask = torch.sigmoid(mask)
        return mask

    def get_tokenizer(self):
        return self.text_embedder.tokenizer


if __name__ == '__main__':
    input = torch.randn(2, 1, 1001, 513).to('cuda')
    caption = ['[CLS] a bird is singing', '[CLS] a dog is barking']
    model = nn.DataParallel(LASSNet().to('cuda'))
    ckpt_path = '/vol/research/MachineAudition_CVSSP/xl01061/LASS_develop/workspace/Final-LTSS-MAE/chkpt/best_for_paper.pt'
    checkpoint = torch.load(ckpt_path)
    import pdb; pdb.set_trace()
    model.load_state_dict(checkpoint['model'])
    print(model(input, caption).shape)