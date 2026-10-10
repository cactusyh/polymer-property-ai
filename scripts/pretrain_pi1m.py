#!/usr/bin/env python
from __future__ import annotations
import argparse, json, math, os, random, sys
from pathlib import Path
import yaml, torch, pandas as pd
from torch import nn
from torch.utils.data import DataLoader, Dataset
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.data.psmiles_tokenizer import tokenize_psmiles
from src.models.psmiles_bert import PSmilesBert, PSmilesBertConfig, MLMHead
SPECIAL=['<PAD>','<MASK>','<UNK>','<CLS>','<SEP>']
def vocab_from_pi(path):
    import collections
    df=pd.read_parquet(path,columns=['raw_psmiles']); c=collections.Counter()
    for s in df.raw_psmiles.astype(str): c.update(tokenize_psmiles(s))
    toks=SPECIAL+sorted([t for t in c if t not in SPECIAL]); return {t:i for i,t in enumerate(toks)}
def encode(s,vocab,maxlen):
    ids=[vocab['<CLS>']]+[vocab.get(t,vocab['<UNK>']) for t in tokenize_psmiles(str(s))]+[vocab['<SEP>']]
    ids=ids[:maxlen]; mask=[1]*len(ids); ids+= [0]*(maxlen-len(ids)); mask += [0]*(maxlen-len(mask)); return torch.tensor(ids),torch.tensor(mask)
class SmilesDS(Dataset):
    def __init__(self,path,vocab,maxlen,limit=None): self.df=pd.read_parquet(path,columns=['raw_psmiles','split']).query('split=="train"').head(limit); self.vocab=vocab; self.maxlen=maxlen
    def __len__(self): return len(self.df)
    def __getitem__(self,i): return encode(self.df.iloc[i].raw_psmiles,self.vocab,self.maxlen)
def mask_batch(ids,vocab,p=0.15):
    labels=torch.full_like(ids,-100); special=torch.tensor([vocab[x] for x in SPECIAL],device=ids.device)
    can=(ids!=0)&~torch.isin(ids,special); m=(torch.rand_like(ids.float())<p)&can; labels[m]=ids[m]; ids=ids.clone(); ids[m]=vocab['<MASK>']; return ids,labels
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/pretrain_pi1m.yaml'); ap.add_argument('--max-steps',type=int); ap.add_argument('--limit',type=int); args=ap.parse_args(); cfg=yaml.safe_load(open(args.config)); torch.manual_seed(cfg['seed'])
    vocab=vocab_from_pi(cfg['train_path']); out=Path(cfg['output_dir']); out.mkdir(parents=True,exist_ok=True); json.dump(vocab,open(out/'vocab.json','w'))
    mc=cfg['model']; model=PSmilesBert(PSmilesBertConfig(vocab_size=len(vocab),**mc)); head=MLMHead(mc['hidden_size'],len(vocab)); opt=torch.optim.AdamW(list(model.parameters())+list(head.parameters()),lr=cfg['training']['lr'],weight_decay=cfg['training'].get('weight_decay',0.01))
    ds=SmilesDS(cfg['train_path'],vocab,mc['max_sequence_length'],args.limit); dl=DataLoader(ds,batch_size=cfg['training']['batch_size'],shuffle=True)
    rows=[]; step=0; model.train(); head.train()
    for ep in range(cfg['training']['epochs']):
      for ids,mask in dl:
        ids_m,labels=mask_batch(ids,vocab,cfg['training']['mask_probability']); x,_=model(ids_m,mask); logits=head(x); loss=nn.functional.cross_entropy(logits.view(-1,len(vocab)),labels.view(-1),ignore_index=-100); loss.backward(); nn.utils.clip_grad_norm_(list(model.parameters())+list(head.parameters()),cfg['training']['max_grad_norm']); opt.step(); opt.zero_grad(); step+=1
        pred=logits.argmax(-1); masked=labels!=-100; acc=float((pred[masked]==labels[masked]).float().mean()) if masked.any() else math.nan; rows.append({'step':step,'epoch':ep,'train_mlm_loss':float(loss),'masked_token_accuracy':acc})
        if args.max_steps and step>=args.max_steps: break
      if args.max_steps and step>=args.max_steps: break
    pd.DataFrame(rows).to_csv(out/'training_curves.csv',index=False); torch.save({'model':model.state_dict(),'head':head.state_dict(),'config':mc,'vocab':vocab},out/'best.pt'); print({'steps':step,'output':str(out)})
if __name__=='__main__': main()
