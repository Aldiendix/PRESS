import numpy as np,pandas as pd,hdbscan,umap,glob,sys,warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.neighbors import NearestNeighbors
sc=lambda g,p:(max(0,ari(g,p))+nmi(g,p))/2
tot={}
for f in sorted(glob.glob('data/round_0073/*.parquet')):
    df=pd.read_parquet(f); gt=df.label.values
    E=np.load('cache/'+f.split('/')[-1].replace('.parquet','.mpnet.npy')); E/=np.linalg.norm(E,axis=1,keepdims=True)
    out={}
    # (b) competitor-style clustering on teacher embeddings
    L=linkage(E,'average','cosine'); d=NearestNeighbors(n_neighbors=15,metric='cosine').fit(E).kneighbors(E)[0][:,-1]
    for k in (40,100):
        for fr in (0,.15,.25):
            l=fcluster(L,k,'maxclust'); idx=np.argsort(-d)[:int(len(l)*fr)]
            l1=l.copy(); l1[idx]=10**6+np.arange(len(idx)); l2=l.copy(); l2[idx]=-1
            out['agg k%d fr%.2f single'%(k,fr)]=sc(gt,l1); out['agg k%d fr%.2f noise'%(k,fr)]=sc(gt,l2)
    for seed in (1,2):
        U=umap.UMAP(n_neighbors=15,n_components=5,min_dist=0.0,metric='cosine',random_state=seed).fit_transform(E)
        for mcs,ms in ((25,5),(25,10),(30,5),(20,5),(25,None),(30,10)):
            l=hdbscan.HDBSCAN(min_cluster_size=mcs,min_samples=ms).fit_predict(U)
            out.setdefault('umap+hdb mcs%d ms%s'%(mcs,ms),[]).append(sc(gt,l))
    for k,v in out.items(): tot.setdefault(k,[]).append(float(np.mean(v)))
    print(f[-30:],flush=True)
for k,v in tot.items(): print('%-28s'%k,' '.join('%.3f'%x for x in v),'| mean %.4f'%np.mean(v))
