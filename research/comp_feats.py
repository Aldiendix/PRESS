"""Capture the feature matrix the reference solution feeds into linkage(); cluster it with real UMAP+HDBSCAN."""
import numpy as np,pandas as pd,hdbscan,umap,glob,sys,warnings,importlib.util,os
warnings.filterwarnings('ignore')
from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi
sc=lambda g,p:(max(0,ari(g,p))+nmi(g,p))/2
spec=importlib.util.spec_from_file_location('ref','reference/r73_5C55Guoe_v1_0.4749.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
orig=m.p; cap={}
def hook(G,*a,**k): cap['G']=np.array(G); return orig(G,*a,**k)
m.p=hook
tot={}
for f in sorted(glob.glob('data/round_%04d/*.parquet'%int(sys.argv[1]))):
    df=pd.read_parquet(f); gt=df.label.values
    pred=np.array(m.cluster_texts(df.text.tolist())); G=cap['G']
    tot.setdefault('reference as-is',[]).append(sc(gt,pred))
    if len(G)!=len(gt): gt=gt[pred!=-1]; pred=pred[pred!=-1]
    np.save('cache/'+os.path.basename(f).replace('.parquet','.compG.npy'),G)

    for nn in (15,30):
        U=umap.UMAP(n_neighbors=nn,n_components=5,min_dist=0.0,metric='cosine',random_state=1).fit_transform(G)
        for mcs,ms in ((30,5),(30,10),(50,10)):
            l=hdbscan.HDBSCAN(min_cluster_size=mcs,min_samples=ms).fit_predict(U)
            tot.setdefault('G umap nn%d mcs%d ms%d'%(nn,mcs,ms),[]).append(sc(gt,l))
            l2=l.copy(); n=l2==-1; l2[n]=10**6+np.arange(n.sum())
            tot.setdefault('G umap nn%d mcs%d ms%d noise->single'%(nn,mcs,ms),[]).append(sc(gt,l2))
    print(f[-30:],G.shape,flush=True)
for k,v in sorted(tot.items(),key=lambda kv:-np.mean(kv[1])): print('%-40s'%k,' '.join('%.3f'%x for x in v),'| mean %.4f'%np.mean(v))
