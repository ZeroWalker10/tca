import numpy as np
import pdb

class TranslationMap:
    def __init__(self, nodes=None, trans_lib=None, T=0.2, depth_T={}): 
        self.trans_lib = trans_lib
        self.T = T
        if nodes is not None:
            self.nodes = nodes
            self._map_construction()

            collected_nodes = []
            depth_collected_nodes = {}
            for d, ns in nodes.items():
                depth_nodes = []
                for tran_seq, tran_score in ns.items():
                    depth_nodes.append((tran_seq, tran_score))
                collected_nodes.extend(depth_nodes)
                depth_collected_nodes[d] = depth_nodes

            sorted_collected_nodes = sorted(collected_nodes, key=lambda x: x[1], reverse=True)
            self.sorted_tran_seqs = [x[0] for x in sorted_collected_nodes]
            self.sorted_tran_scores = [x[1] for x in sorted_collected_nodes]
            self.sorted_tran_probs = np.exp(np.array(self.sorted_tran_scores, dtype=np.float32) / self.T)
            self.sorted_tran_probs = self.sorted_tran_probs / self.sorted_tran_probs.sum()

            self.depth_sorted_tran_seqs = {}
            self.depth_sorted_tran_scores = {}
            self.depth_sorted_tran_probs = {}
            self.depths = tuple(self.nodes.keys())
            self.dindex = 0
            if tuple(sorted(depth_T.keys())) == tuple(sorted(nodes.keys())):
                for d, collected_nodes in depth_collected_nodes.items():
                    depth_sorted_collected_nodes = sorted(collected_nodes, key=lambda x: x[1], reverse=True)
                    self.depth_sorted_tran_seqs[d] = [x[0] for x in depth_sorted_collected_nodes]
                    self.depth_sorted_tran_scores[d] = [x[1] for x in depth_sorted_collected_nodes]
                    max_score = np.max(self.depth_sorted_tran_scores[d])
                    self.depth_sorted_tran_probs[d] = \
                            np.exp(np.array(self.depth_sorted_tran_scores[d] - max_score, dtype=np.float32) / depth_T[d])
                    self.depth_sorted_tran_probs[d] = self.depth_sorted_tran_probs[d] / self.depth_sorted_tran_probs[d].sum()

            self.start_node = -1
            self.end_node = len(self.trans_lib)

    def _presentative_score(self, node):
        max_score = 0.0
        for d, ns in self.nodes.items():
            for n, score in ns.items():
                if len(n) >= len(node) and n[:len(node)] == node:
                    max_score = max(max_score, score)
        return max_score
    
    def _to_prob(self, scores):
        results = np.exp(scores / self.T)
        return results / results.sum()
    
    def _map_construction(self):
        self.map = {}
        self.collected_nodes = {}
        for d, nodes in self.nodes.items():
            self.collected_nodes.update(nodes)
            prev_nodes = set(n[:-1] for n in nodes.keys())
            for pnode in prev_nodes:
                tran_scores = np.ones(len(self.trans_lib)+1, dtype=np.float32) * -np.inf
                for cnode in nodes.keys():
                    if pnode == cnode[:-1]:
                        tran_scores[cnode[-1]] = self._presentative_score(cnode) 

                if pnode in self.collected_nodes:
                    tran_scores[-1] = self.collected_nodes[pnode] # stop at this node

                tran_probs = self._to_prob(tran_scores)
                self.map[pnode] = tran_probs
                self.have_chosen = []

    def random_translate(self, inputs, r=1, have_list=[]):
        tran_seq = np.random.choice(np.arange(len(self.trans_lib)), size=(r,))

        new_inputs = inputs.clone()
        # (x, x, ...) 
        for tran_i in tran_seq:
            new_inputs = self.trans_lib[tran_i](new_inputs)
        return new_inputs, tran_seq 

    def _tran_seq_len(self, tran_seq):
        exclude = 0
        if self.start_node in tran_seq:
            exclude = exclude + 1
        if self.end_node in tran_seq:
            exclude = exclude + 1

        return len(tran_seq) - exclude

    def update_dindex(self):
        self.dindex = (self.dindex + 1) % len(self.depths)

    def set_depth(self, depth):
        self.dindex = self.depths.index(depth)

    def pick_translation(self, mode=None, r=1, index=None, d=1):
        if mode == 'random':
            tran_seq = np.random.choice(np.arange(len(self.trans_lib)), size=(r,), replace=False)
            tran_seq = (-1,) + tran_seq
        elif mode == 'index':
            tran_seq = self.sorted_tran_seqs[index]
            tran_seq = (-1,) + tran_seq
        elif mode == 'priority':
            choice = np.random.choice(np.arange(len(self.sorted_tran_seqs)), p=self.sorted_tran_probs)
            tran_seq = self.sorted_tran_seqs[choice]
            tran_seq = (-1,) + tran_seq
        elif mode == 'hamburger':
            choice = np.random.choice(np.arange(len(self.depth_sorted_tran_seqs[self.depths[self.dindex]])), 
                p=self.depth_sorted_tran_probs[self.depths[self.dindex]])
            tran_seq = self.depth_sorted_tran_seqs[self.depths[self.dindex]][choice]
            tran_seq = (-1,) + tran_seq
        elif mode == 'depth':
            tran_seq = (-1,)
            while self._tran_seq_len(tran_seq) != d: 
                tran_seq = (-1,)
                next_node = np.random.choice(np.arange(len(self.trans_lib)+1), p=self.map[tran_seq])
                tran_seq = tran_seq + (next_node,)
                while tran_seq in self.map:
                    next_node = np.random.choice(np.arange(len(self.trans_lib)+1), p=self.map[tran_seq])
                    tran_seq = tran_seq + (next_node,)
        else:
            tran_seq = (-1,)
            next_node = np.random.choice(np.arange(len(self.trans_lib)+1), p=self.map[tran_seq])
            tran_seq = tran_seq + (next_node,)
            while tran_seq in self.map:
                next_node = np.random.choice(np.arange(len(self.trans_lib)+1), p=self.map[tran_seq])
                tran_seq = tran_seq + (next_node,)

        return tran_seq

    def depth_translate(self, inputs, d=1, have_list=[]): 
        tran_seq = self.pick_translation(mode='depth', d=d)

        new_inputs = inputs.clone()
        # (-1, x, x, ...) 
        for tran_i in tran_seq[1:]:
            if tran_i < len(self.trans_lib):
                new_inputs = self.trans_lib[tran_i](new_inputs)
        return new_inputs, tran_seq 

    def hamburger_translate(self, inputs, have_list=[]):
        tran_seq = self.pick_translation(mode='hamburger')

        new_inputs = inputs.clone()
        # (-1, x, x, ...) 
        for tran_i in tran_seq[1:]:
            if tran_i < len(self.trans_lib):
            	new_inputs = self.trans_lib[tran_i](new_inputs)
        return new_inputs, tran_seq 

    def priority_translate(self, inputs, have_list=[]):
        tran_seq = self.pick_translation(mode='priority')

        new_inputs = inputs.clone()
        # (-1, x, x, ...) 
        for tran_i in tran_seq[1:]:
            if tran_i < len(self.trans_lib):
            	new_inputs = self.trans_lib[tran_i](new_inputs)
        return new_inputs, tran_seq 

    def index_translate(self, inputs, index, have_list=[]):
        tran_seq = self.pick_translation(mode='index', index=index)

        new_inputs = inputs.clone()
        # (-1, x, x, ...) 
        for tran_i in tran_seq[1:]:
            if tran_i < len(self.trans_lib):
            	new_inputs = self.trans_lib[tran_i](new_inputs)
        return new_inputs, tran_seq 

    def translate(self, inputs, tran_seq=None):
        if tran_seq is None:
            tran_seq = self.pick_translation(mode=None)
                
        new_inputs = inputs.clone()
        # (-1, x, x, ...) 
        for tran_i in tran_seq[1:]:
            if tran_i < len(self.trans_lib):
            	new_inputs = self.trans_lib[tran_i](new_inputs)
        return new_inputs, tran_seq 
