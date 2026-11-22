"""Cross-view stress check (D-012). Run: python -m models.stress_crossview"""
# Cross-view stress: known queries whose flank is absent from the gallery.
import numpy as np, pandas as pd
from reid.data import load, NEW
from reid.split import make_split
from reid.metric import map5
from models.baseline_knn import make_predict as knn
from models.metric_learning import make_predict as ml
train,_,X,_=load()
SIDE={'left':'L','frontleft':'L','backleft':'L','right':'R','frontright':'R','backright':'R'}
side=train.viewpoint.map(SIDE).fillna('O').values
for f in range(2):
    g,q,t=make_split(f)
    rng=np.random.default_rng(f)
    gal=set(g.tolist())
    # for gallery identities with both flanks, move one whole flank of some ids into queries
    gi=pd.Series(list(gal)); ids=train.identity.values
    moved=[]
    for idn,grp in train.iloc[list(gal)].groupby('identity'):
        s=side[grp.index]
        if (s=='L').any() and (s=='R').any() and rng.random()<0.5:
            fl=rng.choice(['L','R']); moved+=list(grp.index[s==fl])
    moved=np.array(moved); g2=np.array(sorted(gal-set(moved.tolist())))
    q2=np.r_[q,moved]; t2=np.r_[t,ids[moved]].astype(object)
    G=train.iloc[g2].reset_index(drop=True); Q=train.iloc[q2].drop(columns='identity').reset_index(drop=True)
    for name,p in [('knn T.45',knn({'hyena':.45,'leopard':.45})),('knn T.35',knn({'hyena':.35,'leopard':.35})),('cos+lda',ml(dict(lda=128,shrink=0.3),{'hyena':.465,'leopard':.509}))]:
        P=p(G,X[g2],Q,X[q2]); cv=np.r_[np.zeros(len(q),bool),np.ones(len(moved),bool)]
        print(f,name,'all %.4f'%map5(P,t2),'| orig queries %.4f'%map5([P[i] for i in np.where(~cv)[0]],t2[~cv]),'| cross-view known %.4f (n=%d)'%(map5([P[i] for i in np.where(cv)[0]],t2[cv]),cv.sum()),'| new-first on cross-view %.1f%%'%(100*np.mean([P[i][0]==NEW for i in np.where(cv)[0]])))
