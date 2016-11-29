# coding=utf-8
from __future__ import unicode_literals
import re
import sys
import codecs
import os.path as osp
import numpy as np
import networkx as nx
import random
from gensim.models.word2vec import LineSentence,Word2Vec
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from collections import defaultdict
from six.moves import zip_longest
random.seed(2017)
np.random.seed(2017)


class ThinGraph(defaultdict):
    """
    Efficient implementation of nx.DiGraph for reduce memory used.
    """
    def __init__(self):
        super(ThinGraph, self).__init__(dict)

    def order(self):
        return len(self)

    def nodes(self):
        return self.iterkeys()

    def neighbors(self, u):
        return list(self[u])

    def has_node(self, u):
        return u in self

    def has_edge(self,u,v):
        return self.has_node(u) and v in self[u]

    def edges(self):
        for u in self.nodes():
            for v in self.neighbors(u):
                yield (u, v, self[u][v])

    def add_edges_from(self,edges):
        for e in edges:
            self.add_edge(*e)

    def add_node(self,u):
        if u not in self:
            self[u]={}

    def add_edge(self, u, v, d=1.0):
        if u != v:
            self[u][v] = d

    def make_undirected(self):
        for u,v,d in self.edges():
            self.add_edge(v,u,d)

    def load_adjacencylist(self, path):
        with open(path,'r') as f:
            for l in f:
                if l and l[0] != "#":
                    vlist=l.strip().split()
                    vlist=map(int,vlist)
                    u=vlist.pop(0)
                    self.add_edges_from([(u, v) for v in vlist])

    def load_edgelist(self,path, func=lambda x: x):
        with open(path, 'r') as f:
            for l in f:
                u, v, d=l.strip().split(',')
                self.add_edge((int(u), int(v), func(float(d))))


class Graph():
    def __init__(self, nx_G, p, q):
        self.G = nx_G
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G

        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    alias_node=self.get_alias_node(cur)
                    if alias_node:
                        walk.append(cur_nbrs[alias_draw(alias_node[0], alias_node[1])])
                    else:
                        walk.append(random.choice(cur_nbrs))
                else:
                    prev = walk[-2]
                    alias_edges=self.get_alias_edge(prev, cur)
                    next = cur_nbrs[alias_draw(alias_edges[0],
                                               alias_edges[1])]
                    walk.append(next)
            else:
                break
        return walk

    def random_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = G.neighbors(cur)
            if len(cur_nbrs) > 0:
                    walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk

    def simulate_walks(self, out_file, num_walks, walk_length,method='random'):
        '''
        Repeatedly simulate random walks from each node.
        method='random' or 'net2vec'
        random: 和deepwalk一样随机选
        net2vec：根据概率选
        '''
        walk_method={'node2vec':self.node2vec_walk,'random':self.random_walk}
        G = self.G
        nodes = list(G.nodes())
        nb_node = len(nodes)
        print 'nodes: {}'.format(nb_node)
        print 'Walk iteration:'
        with open(out_file,'w') as f:
            for walk_iter in range(num_walks):
                random.shuffle(nodes)
                iter_nodes = zip_longest(*[iter(nodes)] * 10000)
                print '\n'+str(walk_iter + 1), '/', str(num_walks)
                for i,ns in enumerate(iter_nodes):
                    sys.stdout.write('{}/{}\r'.format(i*10000,nb_node))
                    sys.stdout.flush()
                    walks=[]
                    for n in ns:
                        if n:
                            walk=walk_method[method](walk_length=walk_length, start_node=n)
                    if len(walk)>2:
                        walks.append(walk)
                    for walk in walks:
                        f.write(' '.join(map(str,walk))+'\n')

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.

        '''
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr] / p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]/ q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def get_alias_node(self,node):
        G=self.G
        unnormalized_probs = [G[node][nbr] for nbr in sorted(G.neighbors(node))]
        if all([p == 1 for p in unnormalized_probs]):
            return None
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
        return alias_setup(normalized_probs)


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K,dtype=np.float32)
    J = np.zeros(K, dtype=np.int32)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def grouper(n, iterable, padvalue=None):
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)


def parse_ulink(f):
    edges = []
    for line in f:
        if line:
            vlist = re.split('\t|\x01', line.strip())
            if len(vlist) < 2: continue
            vlist = map(int, vlist)
            u = vlist.pop(0)
            edges.extend([(v, u) for v in vlist])
    return edges


def load_dc_ulink2(path,chunk_size=1e6):
    """
    load with multi-process
    """
    G=ThinGraph()
    with open(path) as f:
        lines=f.readlines()
        print 'lines: ',len(lines)
    fs=grouper(int(chunk_size), lines)
    for line_group in fs:
        print 'line_group'
        with ProcessPoolExecutor(max_workers=10) as executor:
            for idx, edges in enumerate(executor.map(parse_ulink, grouper(int(1e5), line_group))):
                print idx
                G.add_edges_from(edges)
    return G


def load_dc_ulink(path):
    G = ThinGraph()
    with open(path) as f:
        lines=f.readlines()
        i = 0
        for line in lines:
            i += 1
            if i % 100000 == 0: print i
            vlist = re.split('\t|\x01', line.strip())
            if len(vlist) < 2: continue
            vlist = map(np.int32, vlist)
            u = vlist.pop(0)
            edges = [(v, u, 1.0) for v in vlist]
            G.add_edges_from(edges)


def repost_weight(repost_path,edgelist_path):
    g=ThinGraph()
    with codecs.open(repost_path,'r',encoding='utf8') as f:
        for i,line in enumerate(f):
            print i
            line=line.split('\x01')
            u,v=int(line[1]),int(line[2])
            if g.has_edge(u,v):
                g[u][v] += 1
            else:
                g.add_edge(u,v,1.0)
    with open(edgelist_path,'w') as f:
        for u,v,d in g.edges():
            f.write('{},{},{}\n'.format(u,v,d))


if __name__=='__main__':
    import cPickle

    # 使用好友关系构建图，这个图和deepwalk的区别在于是有向图，而且可以保存连接权重，内存消耗稍大一些，默认权重为1
    g = load_dc_ulink('weibo_dc_parse2015_link_filter')
    # g.make_undirected() 转化为无向图

    # 增加发生转发的边的权重
    # edgelist=osp.join(cfg.data_dir,'edgelist.csv')
    # repost_weight(cfg.train_repost_file,edgelist)
    # g.load_edgelist(edgelist,lambda x: 1+np.log10(1+x))

    # 当p=1,q=1且method='random'的时候和deepwalk一致，只是这里用的是有向图。method改为node2vec并修改p,q值，可以使用node2vec的随机游走采样方法
    corpus_file = 'walks.csv'
    G = Graph(g, p=1.0, q=1.0)
    G.simulate_walks(out_file=corpus_file, num_walks=2, walk_length=10,method='random')

    print 'simulate_walks finshed!\n start traing...'
    corpus = LineSentence(corpus_file)
    model = Word2Vec(corpus, size=64, window=3, min_count=1, sg=1, workers=cpu_count()-5, iter=2,sorted_vocab=False)
    model.save_word2vec_format('uid2vec.bin',binary=True)

    # uids = cPickle.load(open('train_uids.pkl'))
    # uids = np.unique(uids)
    # model=Word2Vec.load_word2vec_format('uid2vec.bin',binary=True,limit=10000)
    # # u_embed=np.zeros((len(uids),32),dtype='float32')
    # with open('u_embed.csv','w') as f:
    #     for i,u in enumerate(uids):
    #         u_embed=model[str(u)].tolist()
    #         f.write(str(u)+','+','.join(map(str,u_embed))+'\n')



