import numpy as np,pandas as pd,hdbscan,umap,glob,sys,itertools,warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi
f=sorted(glob.glob('data/round_0073/*.parquet'))[int(sys.argv[1])]
df=pd.read_parquet(f); gt=df.label.values
E=np.load('cache/'+f.split('/')[-1].replace('.parquet','.mpnet.npy'))
print(f[-30:],'gt k',gt.max()+1,'noise',(gt==-1).sum(),flush=True)
for nn,nc,seed in [(15,5,42),(15,5,0),(15,5,1),(30,5,42),(15,10,42),(10,5,42),(50,5,42),(15,2,42),(15,3,42)]:
    U=umap.UMAP(n_neighbors=nn,n_components=nc,min_dist=0.0,metric='cosine',random_state=seed).fit_transform(E)
    res=[]
    for mcs in (15,20,25,30):
        for ms in (0,5,10):
            for meth in ('eom','leaf'):
                l=hdbscan.HDBSCAN(min_cluster_size=mcs,min_samples=ms or None,cluster_selection_method=meth).fit_predict(U)
                res.append((ari(gt,l),nmi(gt,l),mcs,ms,meth,l.max()+1,(l==-1).sum()))
    res.sort(reverse=True)
    print('umap nn%d nc%d seed%d:'%(nn,nc,seed),' | '.join('ari %.3f nmi %.3f mcs%d ms%d %s k%d n%d'%r for r in res[:3]),flush=True)
