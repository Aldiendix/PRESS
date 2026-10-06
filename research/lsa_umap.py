import numpy as np,pandas as pd,hdbscan,umap,glob,sys,warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from scipy.sparse import hstack
sc=lambda g,p:(max(0,ari(g,p))+nmi(g,p))/2
tot={}
for f in sorted(glob.glob('data/round_0073/*.parquet')):
    df=pd.read_parquet(f); gt=df.label.values; T=df.text.tolist()
    A=normalize(TfidfVectorizer(max_features=100000,min_df=2,max_df=.5,sublinear_tf=True,stop_words='english',ngram_range=(1,2),dtype=np.float32).fit_transform(T))
    B=normalize(TfidfVectorizer(max_features=100000,min_df=2,max_df=.5,sublinear_tf=True,analyzer='char_wb',ngram_range=(3,5),dtype=np.float32).fit_transform(T))
    for name,X in (('word',A),('word+char',hstack([A,B]).tocsr())):
        for dim in (64,192):
            Z=normalize(TruncatedSVD(dim,random_state=0).fit_transform(X))
            for nn in (15,30):
                U=umap.UMAP(n_neighbors=nn,n_components=5,min_dist=0.0,metric='cosine',random_state=1).fit_transform(Z)
                for mcs,ms in ((30,5),(30,10),(50,10)):
                    l=hdbscan.HDBSCAN(min_cluster_size=mcs,min_samples=ms).fit_predict(U)
                    tot.setdefault('%s svd%d nn%d mcs%d ms%d'%(name,dim,nn,mcs,ms),[]).append((sc(gt,l),l.max()+1,(l==-1).sum()))
    print(f[-30:],flush=True)
for k,v in sorted(tot.items(),key=lambda kv:-np.mean([x[0] for x in kv[1]])): print('%-34s'%k,' '.join('%.3f(k%d,n%d)'%x for x in v),'| mean %.4f'%np.mean([x[0] for x in v]))
