import dgl
import torch as th

u, v = th.tensor([0,0,0,1]), th.tensor([1,2,3,3])
g = dgl.graph((u, v))
print('g', g)
print('nodes', g.nodes())
print('edges',g.edges())