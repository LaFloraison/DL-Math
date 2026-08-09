import numpy as np
def relu(h): return np.maximum(h, 0.0)
def init_block(depth, width, scale, seed=0):
    rng = np.random.default_rng(seed)
    return [(rng.normal(0, scale, (width,width)), np.zeros(width), rng.normal(0, scale, (width,width)), np.zeros(width)) for _ in range(depth)]
def forward(x, blocks):
    recs=[]; h=x
    for (W1,b1,W2,b2) in blocks:
        z1=relu(h@W1.T+b1); z2=z1@W2.T+b2; F=relu(z2)
        recs.append((h,z1,z2)); h=F+h
    return h, recs
def train(X,y,depth,width,scale,residual,steps,lr,seed=0):
    rng=np.random.default_rng(seed)
    blocks0=init_block(depth,width,scale,seed)
    blocks=[(W1.copy(),b1.copy(),W2.copy(),b2.copy()) for (W1,b1,W2,b2) in blocks0]
    w=rng.normal(0,0.05,width); curve=[]
    for t in range(steps):
        hL,recs=forward(X,blocks)
        pred=np.tanh(hL@w); dloss=2.0*(pred-y)*(1.0-pred**2)
        w-=lr*(hL.T@dloss); delta=dloss[:,None]*w[None,:]
        for l in range(depth-1,-1,-1):
            hp,z1,z2=recs[l]; W1,b1,W2,b2=blocks[l]
            dF=delta if residual else delta*(z2>0).astype(float)
            dz2=dF*(z2>0).astype(float); gW2=dz2.T@z1
            dz1=(dz2@W2)*(z1>0).astype(float); gW1=dz1.T@hp
            d_hp=dz1@W1 + (delta if residual else 0.0)
            W1-=lr*gW1; W2-=lr*gW2; b1-=lr*dz1.sum(0); b2-=lr*dz2.sum(0)
            blocks[l]=(W1,b1,W2,b2); delta=d_hp
        if t%3000==0: curve.append(float(np.mean((pred-y)**2)))
    return [f"{c:.3f}" for c in curve], np.linalg.norm(blocks[0][0]-blocks0[0][0])
rng=np.random.default_rng(3)
X=rng.normal(0,1,(128,16))
y=np.sign(np.sin(X[:,0]*X[:,1])+X[:,2]**2-X[:,3]); y=(y>0).astype(float)
for scale,lr,steps in ((0.1,0.003,15000),(0.1,0.001,20000)):
    print(f"=== depth=15 scale={scale} lr={lr} steps={steps} ===")
    for residual in (False,True):
        curve,dW0=train(X,y,15,16,scale,residual,steps,lr)
        print(f"  residual={residual}: loss {curve} | dW0={dW0:.2e}")
