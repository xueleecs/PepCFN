
from transformers import BertModel, BertTokenizer
import re, torch
import torch.nn as nn
import torch.nn.functional as F
import esm
from functools import partial
import numpy as np
from train.SperXonv1d import SpectralConv1d



def get_plm_reps(seq, model, converter):
    # seq = ''.join([RESTYPE_3to1(i) for i in list(df['resname'])])
    batch_tokens = converter([("_", seq)])[2]  # [label, sequence]
    with torch.no_grad():
        results = model(batch_tokens.cuda(), repr_layers=[33], return_contacts=True)  # Extract per-residue representations
    token_reps = results["representations"][33][0].detach()         #[:, 1:-1]         # the head and tail tokens are placeholder
    return token_reps


def freeze_bert(self):
    for name, child in self.bert.named_children():
        for param in child.parameters():
            param.requires_grad = False

class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()

        global max_len, d_model, device
        # max_len = config.max_len
        d_model = 1280#1024
        device = torch.device("cuda" if config.cuda else "cpu")
        
        self.tokenizer = BertTokenizer.from_pretrained('/home/bio-3090ti/BenzcodeL/NO.6/prot_bert_bfd', do_lower_case=False)
        self.bert = BertModel.from_pretrained("/home/bio-3090ti/BenzcodeL/NO.6/prot_bert_bfd")

        self.modelesm,alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self.batch_converter = alphabet.get_batch_converter()


        # freeze_bert(self)
        self.cnn = nn.Sequential(nn.Conv1d(in_channels=1280,
                                        out_channels=1280,
                                        kernel_size=13,
                                           stride=1,
                                           padding=6),
                              nn.ReLU(),
                              # nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
                                 )
        self.conv1d = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=512, kernel_size=(1, 1024), stride=(1, 1),
                                    padding=(0, 0)),
                                  nn.ReLU())

        self.lstm = nn.LSTM(d_model, d_model // 2, num_layers=2, batch_first=True, bidirectional=True)
        self.GRU = nn.GRU(d_model, d_model // 2, num_layers=2, batch_first=True, bidirectional=True)
        self.gru = nn.GRU(d_model, d_model, num_layers=2, batch_first=True)

        self.Tcnn1 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=1),nn.ReLU())

        self.Tpool1 = nn.MaxPool1d(kernel_size=2) 

        self.Tcnn2 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, padding=2, stride=1),nn.ReLU())

        self.Tpool2 = nn.MaxPool1d(kernel_size=2) 

        self.Tcnn3 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=7, padding=3, stride=1),nn.ReLU())#padding = (kernel_size - 1) // 2
        self.Tpool3 = nn.MaxPool1d(kernel_size=4)

        self.Tcnn4 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=9, padding=4, stride=1),nn.ReLU())
        self.Tpool4 = nn.MaxPool1d(kernel_size=4)

        self.dconv1d = torch.nn.ConvTranspose1d(512, 1024, kernel_size=2, stride=2,bias=False)

        self.dconv1d1 = torch.nn.ConvTranspose1d(256, 512, kernel_size=2, stride=2,bias=False)

        self.avgpool = nn.AvgPool2d(2)
        self.q = nn.Parameter(torch.empty(d_model,))
        # self.q.data.fill_(1)
        self.block1 = nn.Sequential(
 
            nn.Linear(2304, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        self.block2 = nn.Sequential(

            nn.ReLU(),
            nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

        self.width=1024
        self.modes1=10
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(1280, self.width) # input channel is 2: (a(x), x)
        self.fc00 = nn.Linear(1024, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv4 = SpectralConv1d(self.width, self.width, self.modes1)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.w4 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 256)

    def FNO1d(self,x):
        
        x0=x
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # print(x.shape)
        
        # x = F.pad(x, [0, self.padding]) # pad the domain if input is non-periodic
        X=x.shape
        xx=X[2]
        xxx=X[1]
        if xx>1024:
            x=x[:,:,:1024]
        else:
            x=F.pad(x,[0,1024-xx])

        ########################初始残差#########################

        x1 = self.conv0(x)
        x2 = self.w0(x)
        xx0 = x1 + x2
        xx0 = F.gelu(xx0)

        y1=x+xx0
        x1 = self.conv1(y1)
        x2 = self.w1(y1)
        xx1 = x1 + x2
        xx1 = F.gelu(xx1)

        y2=x+xx1
        x1 = self.conv2(y2)
        x2 = self.w2(y2)
        xx2 = x1 + x2
        xx2 = F.gelu(xx2)

        y3=x+xx2
        x1 = self.conv3(y3)
        x2 = self.w3(y3)
        xx3 = x1 + x2
        xx3 = F.gelu(xx3)

        x=x+xx3

        x=x[:,:,:xx]
        x = x.permute(0, 2, 1)
        x0=x0[:,:,xxx:]
        x=torch.cat([x,x0],dim=2)

        return x

    
    def FNO2d(self,x):
        x0=x
        x = self.fc00(x)
        x = x.permute(0, 2, 1)
        # print(x.shape)
        

        ##############################################
        
        X=x.shape
        xx=X[2]
        xxx=X[1]
        if xx>1024:
            x=x[:,:,:1024]
        else:
            x=F.pad(x,[0,1024-xx])

        ########################初始残差#########################

        x1 = self.conv0(x)
        x2 = self.w0(x)
        xx0 = x1 + x2
        xx0 = F.gelu(xx0)

        y1=x+xx0
        x1 = self.conv1(y1)
        x2 = self.w1(y1)
        xx1 = x1 + x2
        xx1 = F.gelu(xx1)

        y2=x+xx1
        x1 = self.conv2(y2)
        x2 = self.w2(y2)
        xx2 = x1 + x2
        xx2 = F.gelu(xx2)

        y3=x+xx2
        x1 = self.conv3(y3)
        x2 = self.w3(y3)
        xx3 = x1 + x2
        xx3 = F.gelu(xx3)

        x=x+xx3
        x=x[:,:,:xx]

        x = x.permute(0, 2, 1)

        x0=x0[:,:,xxx:]
        x=torch.cat([x,x0],dim=2)

        return x


    def attention(self, input, q):

        att_weights = F.softmax(q, 0)
        output = torch.mul(att_weights, input)
        return output
    
    def forward(self, input_seq):

        input_seq = ' '.join(input_seq)
        input_seq = re.sub(r"[UZOB]", "X", input_seq)

        #Bert
        encoded_input = self.tokenizer(input_seq, return_tensors='pt')
        for key in encoded_input:
            encoded_input[key] = encoded_input[key].cuda()
        outputt = self.bert(**encoded_input)
        bb=outputt[0] #LX-[1,123,1024]

        #ESM
        output=get_plm_reps(input_seq, self.modelesm, self.batch_converter)#[140,1280]
        ee = output.unsqueeze(0)#LX-[1,123,1280]


        representationESM=self.FNO1d(ee)
        representationbert=self.FNO2d(bb)
        
        representationESM=representationESM+ee
        representationbert=representationbert+bb

        representation1 = representationESM.view(-1, 1280)
        representation2 = representationbert.view(-1, 1024)

        representation = torch.cat([representation1,representation2],dim=1)
        representation = self.block1(representation)
        
        return representation
    def get_logits(self, x):
        with torch.no_grad():
            output = self.forward(x)
        logits = self.block2(output)
        return logits
