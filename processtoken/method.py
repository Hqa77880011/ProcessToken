import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):return x.view_as(x)
    @staticmethod
    def backward(ctx,g):return -g


class Model(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;w=c.get('width',32);v=c.get('variables',6)
        self.variable=nn.Embedding(v,w);self.stage=nn.Embedding(c.get('stages',8),w);self.recipe=nn.Embedding(c.get('recipes',4),w)
        self.value=nn.Linear(3,w);self.decay=nn.Linear(w,w);self.key=nn.Linear(w,w);self.jump=nn.Linear(w,w);self.memory=nn.Linear(3*w,w)
        self.process=nn.Linear(2*w,w);self.tool=nn.Linear(2*w,w);self.head=nn.Linear(w,c.get('classes',3));self.tool_head=nn.Linear(w,c.get('tools',4));self.reconstruct=nn.Linear(w,16)
        self.register_buffer('graph',torch.tensor(c.get('graph',np.eye(v).tolist())).bool())
    def forward(self,b):
        x=b['x'].float();all_process=[];all_tool=[];all_reconstruction=[];all_mask=[]
        for events in x:
            visible=events[:,4]<=self.c.get('decision_stage',4);state=events.new_zeros(self.head.in_features);memory=torch.zeros_like(state);previous_time=events.new_tensor(0.);previous_stage=-1;history=torch.zeros(len(self.graph),dtype=torch.bool,device=x.device)
            reconstruction=events.new_zeros(len(events),16);masked=torch.zeros(len(events),dtype=torch.bool,device=x.device);cursor=0
            while cursor<len(events):
                end=cursor+1
                while end<len(events) and bool((events[end,3]==events[cursor,3])&(events[end,4]==events[cursor,4])):end+=1
                group=events[cursor:end];ids=group[:,0].long();stage=group[:,4].long();recipe=group[:,5].long();observed=group[:,2]>0
                supported=(history[None]&(self.graph[:,ids].T|F.one_hot(ids,len(history)).bool())).any(-1)
                if self.c['variant']=='random_mask':supported=torch.ones_like(supported)
                mask=(torch.rand(len(group),device=x.device)<self.c.get('mask_rate',.2))&supported&observed if self.training else torch.zeros_like(observed)
                values=torch.stack([group[:,1]*(~mask),group[:,2]*(~mask),mask.float()],-1)
                delta=(group[0,3]-previous_time).clamp_min(0);token=(self.variable(ids)+self.stage(stage)+self.recipe(recipe)+self.value(values)).mean(0)
                if bool(visible[cursor]):
                    retain=(1-self.key(token).sigmoid())*torch.exp(-F.softplus(self.decay(token))*delta)
                    state=retain*state+self.key(token).sigmoid()*self.jump(token).tanh()
                    if int(stage[0])!=previous_stage and self.c['variant']!='no_stage_memory':
                        gate=self.memory(torch.cat([state,memory,self.stage(stage[0])])).sigmoid();memory=gate*memory+(1-gate)*state
                    reconstruction[cursor:end]=self.reconstruct(state);masked[cursor:end]=mask;history[ids[observed]]=True
                previous_time=group[0,3];previous_stage=int(stage[0]);cursor=end
            latent=torch.cat([state,memory]);all_process.append(self.process(latent));all_tool.append(self.tool(latent));all_reconstruction.append(reconstruction);all_mask.append(masked)
        process=torch.stack(all_process);tool=torch.stack(all_tool);logits=self.head(process)
        return {'probability':logits.softmax(-1),'logits':logits,'process':process,'tool_latent':tool,'adversary':self.tool_head(Reverse.apply(process)),'tool_logits':self.tool_head(tool),'reconstruction':torch.stack(all_reconstruction),'masked':torch.stack(all_mask)}
    def objective(self,b,p):
        loss=F.cross_entropy(p['logits'],b['y'].long());mask=p['masked']
        if mask.any():
            target=((b['x'][...,1].float().clamp(-3,3)+3)/6*15).long();pretext=F.cross_entropy(p['reconstruction'][mask],target[mask]);loss=pretext if self.c.get('stage')=='pretrain' else loss+.1*pretext
        if self.c['variant'] not in {'baseline','no_separation'}:
            a=p['process'];z=p['tool_latent'];a=(a-a.mean(0))/a.std(0,unbiased=False).clamp_min(1e-6);z=(z-z.mean(0))/z.std(0,unbiased=False).clamp_min(1e-6)
            loss=loss+.1*(F.cross_entropy(p['adversary'],b['tool'].long())+F.cross_entropy(p['tool_logits'],b['tool'].long()))+.01*(a.T@z/len(a)).square().mean()
        return loss


def demo(seed,n,c):
    rng=np.random.default_rng(seed);x=np.zeros((n,24,6),np.float32);x[...,0]=np.arange(24)%c.get('variables',6);x[...,1]=rng.normal(size=(n,24));x[...,2]=1;x[...,3]=np.arange(24);x[...,4]=np.arange(24)//6;x[...,5]=0
    return {'x':x,'y':np.argmax(x[:,::6,1][:,:3],1).astype('int64'),'tool':np.arange(n)%c.get('tools',4),'group':np.arange(n)}
