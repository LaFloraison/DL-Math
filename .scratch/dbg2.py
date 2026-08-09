import numpy as np
def relu(h): return np.maximum(h, 0.0)
def init_net(depth, width, seed=0):
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(width)          # He 初始化: 激活不衰减
    return [rng.normal(0, scale, (width, width)) for _ in range(depth)], [np.zeros(width) for _ in range(depth)]
def forward(x, Ws, bs, residual):
    hs=[x]; h=x
    for l,(W,b) in enumerate(zip(Ws,bs)):
        h=relu(h@W.T+b)
        if residual: h=h+hs[-1]
        hs.append(h)
    return hs
def grads(hs, Ws, residual):
    L=len(Ws); delta=np.ones_like(hs[-1]); act=[]; wg=[]
    for l in range(L-1,-1,-1):
        a_prev=hs[l]; a_out=hs[l+1]
        pre_act=a_out-a_prev if residual else a_out
        der=(pre_act>0).astype(float)
        delta_pre=delta*der
        wg.append(np.linalg.norm(delta_pre.T@a_prev))
        g_prev=delta_pre@Ws[l]
        if residual: g_prev=g_prev+delta
        act.append(np.linalg.norm(g_prev)); delta=g_prev
    return act[::-1], wg[::-1]
np.set_printoptions(precision=3, suppress=True)
depth,width=20,20
x=np.random.default_rng(1).normal(0,1,(1,width))
print("||x|| =", np.linalg.norm(x))
for residual in (False,True):
    Ws,bs=init_net(depth,width,seed=0)
    hs=forward(x,Ws,bs,residual)
    act,wg=grads(hs,Ws,residual)
    sel=[0,4,8,12,16,19]
    print(f"residual={residual}: hs_last norm={np.linalg.norm(hs[-1]):.3f}")
    print("  act grad:", [f"{act[i]:.3e}" for i in sel])
    print("  w grad  :", [f"{wg[i]:.3e}" for i in sel])
    print("  one-step dW_bottom:", 0.01*wg[0])
