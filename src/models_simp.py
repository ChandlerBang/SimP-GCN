import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from torch.nn.parameter import Parameter
from utils import sparse_mx_to_torch_sparse_tensor
import scipy.sparse as sp


device = torch.device("cuda:0")


class SimPGCN(nn.Module):
    """
       The model for the single kind of deepgcn blocks.

       The model architecture likes:
       inputlayer(nfeat)--block(nbaselayer, nhid)--...--outputlayer(nclass)--softmax(nclass)
                           |------  nhidlayer  ----|
       The total layer is nhidlayer*nbaselayer + 2.
       All options are configurable.
    """

    def __init__(self,
                 nfeat,
                 nhid,
                 nclass,
                 nhidlayer,
                 dropout,
                 baseblock="mutigcn",
                 inputlayer="gcn",
                 outputlayer="gcn",
                 nbaselayer=0,
                 activation=lambda x: x,
                 withbn=True,
                 withloop=True,
                 aggrmethod="add",
                 mixmode=False, **kwargs):
        """
        Initial function.
        :param nfeat: the input feature dimension.
        :param nhid:  the hidden feature dimension.
        :param nclass: the output feature dimension.
        :param nhidlayer: the number of hidden blocks.
        :param dropout:  the dropout ratio.
        :param baseblock: the baseblock type, can be "mutigcn", "resgcn", "densegcn" and "inceptiongcn".
        :param inputlayer: the input layer type, can be "gcn", "dense", "none".
        :param outputlayer: the input layer type, can be "gcn", "dense".
        :param nbaselayer: the number of layers in one hidden block.
        :param activation: the activation function, default is ReLu.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param aggrmethod: the aggregation function for baseblock, can be "concat" and "add". For "resgcn", the default
                           is "add", for others the default is "concat".
        :param mixmode: enable cpu-gpu mix mode. If true, put the inputlayer to cpu.
        """
        super(SimPGCN, self).__init__()
        self.mixmode = mixmode
        self.dropout = dropout

        print('=== Number of Total Layers is %s ===' % (nhidlayer*nbaselayer + 2))
        if baseblock == "resgcn":
            self.BASEBLOCK = ResGCNBlock
        elif baseblock == "densegcn":
            self.BASEBLOCK = DenseGCNBlock
        elif baseblock == "mutigcn":
            self.BASEBLOCK = MultiLayerGCNBlock
        elif baseblock == "inceptiongcn":
            self.BASEBLOCK = InecptionGCNBlock
        elif baseblock == "tpgcn":
            self.BASEBLOCK = TPGCNBlock
        else:
            raise NotImplementedError("Current baseblock %s is not supported." % (baseblock))
        if inputlayer == "gcn":
            # input gc
            self.ingc = GraphConvolutionBS(nfeat, nhid, activation, withbn, withloop)
            baseblockinput = nhid
        elif inputlayer == "none":
            self.ingc = lambda x: x
            baseblockinput = nfeat
        else:
            self.ingc = Dense(nfeat, nhid, activation)
            baseblockinput = nhid

        outactivation = lambda x: x
        if outputlayer == "gcn":
            self.outgc = GraphConvolutionBS(baseblockinput, nclass, outactivation, withbn, withloop)
        # elif outputlayer ==  "none": #here can not be none
        #    self.outgc = lambda x: x
        else:
            self.outgc = Dense(nhid, nclass, activation)

        # hidden layer
        self.midlayer = nn.ModuleList()
        # Dense is not supported now.
        # for i in xrange(nhidlayer):
        for i in range(nhidlayer):
            gcb = self.BASEBLOCK(in_features=baseblockinput,
                                 out_features=nhid,
                                 nbaselayer=nbaselayer,
                                 withbn=withbn,
                                 withloop=withloop,
                                 activation=activation,
                                 dropout=dropout,
                                 dense=False,
                                 aggrmethod=aggrmethod)
            self.midlayer.append(gcb)
            baseblockinput = gcb.get_outdim()
        # output gc
        outactivation = lambda x: x  # we donot need nonlinear activation here.
        self.outgc = GraphConvolutionBS(baseblockinput, nclass, outactivation, withbn, withloop)

        self.reset_parameters()
        if mixmode:
            self.midlayer = self.midlayer.to(device)
            self.outgc = self.outgc.to(device)

        self.scores = nn.ParameterList()
        self.scores.append(Parameter(torch.FloatTensor(nfeat, 1)))
        for i in range(nhidlayer):
            self.scores.append(Parameter(torch.FloatTensor(nhid, 1)))

        for s in self.scores:
            # s.data.fill_(0)
            stdv = 1. / math.sqrt(s.size(1))
            s.data.uniform_(-stdv, stdv)
            # glorot(s)
            # zeros(self.bias)

        self.bias = nn.ParameterList()
        self.bias.append(Parameter(torch.FloatTensor(1)))
        for i in range(nhidlayer):
            self.bias.append(Parameter(torch.FloatTensor(1)))
        for b in self.bias:
            b.data.fill_(kwargs['bias_init']) # fill in b with postive value to make score s closer to 1 at the beginning

        # self.D_k = Parameter(torch.FloatTensor(kwargs['nnodes'], 1))
        self.D_k = nn.ParameterList()
        self.D_k.append(Parameter(torch.FloatTensor(nfeat, 1)))
        for i in range(nhidlayer):
            self.D_k.append(Parameter(torch.FloatTensor(nhid, 1)))
        for Dk in self.D_k:
            stdv = 1. / math.sqrt(Dk.size(1))
            Dk.data.uniform_(-stdv, stdv)
            # glorot(Dk)
        self.identity = sparse_mx_to_torch_sparse_tensor(sp.eye(kwargs['nnodes'])).to(device)

        self.D_bias = nn.ParameterList()
        self.D_bias.append(Parameter(torch.FloatTensor(1)))
        for i in range(nhidlayer):
            self.D_bias.append(Parameter(torch.FloatTensor(1)))
        for b in self.D_bias:
            b.data.fill_(0) # fill in b with postive value to make score s closer to 1 at the beginning

        self.gamma = kwargs['gamma']

    def reset_parameters(self):
        pass

    def forward(self, fea, adj, adj_knn):
        x, _ = self.myforward(fea, adj, adj_knn)
        return x

    def myforward(self, fea, adj, adj_knn, layer=1.5):
        '''output embedding and log_softmax'''
        gamma = self.gamma

        use_Dk = True
        s_i = torch.sigmoid(fea @ self.scores[0] + self.bias[0])

        if use_Dk:
            Dk_i = (fea @ self.D_k[0] + self.D_bias[0])
            x = (s_i * self.ingc(fea, adj) + (1-s_i) * self.ingc(fea, adj_knn)) + (gamma) * Dk_i * self.ingc(fea, self.identity)
        else:
            x = s_i * self.ingc(fea, adj) + (1-s_i) * self.ingc(fea, adj_knn)

        if layer ==1:
            embedding = x.clone()

        x = F.dropout(x, self.dropout, training=self.training)
        if layer == 1.5:
            embedding = x.clone()

        if self.mixmode:
            x = x.to(device)

        # mid block connections
        # for i in xrange(len(self.midlayer)):
        for i in range(len(self.midlayer)):
            midgc = self.midlayer[i]
            x = midgc(x, adj)
        if layer == -2:
            embedding = x.clone()

        # output, no relu and dropput here.
        s_o = torch.sigmoid(x @ self.scores[-1] + self.bias[-1])
        if use_Dk:
            Dk_o = (x @ self.D_k[-1] + self.D_bias[-1])
            x = (s_o * self.outgc(x, adj) + (1-s_o) * self.outgc(x, adj_knn)) + (gamma) * Dk_o * self.outgc(x, self.identity)
        else:
            x = s_o * self.outgc(x, adj) + (1-s_o) * self.outgc(x, adj_knn)

        if layer == -1:
            embedding = x.clone()
        x = F.log_softmax(x, dim=1)

        self.ss = torch.cat((s_i.view(1,-1), s_o.view(1,-1), gamma*Dk_i.view(1,-1), gamma*Dk_o.view(1,-1)), dim=0)
        return x, embedding


# Modified GCN
class GCNFlatRes(nn.Module):
    """
    (Legacy)
    """
    def __init__(self, nfeat, nhid, nclass, withbn, nreslayer, dropout, mixmode=False):
        super(GCNFlatRes, self).__init__()

        self.nreslayer = nreslayer
        self.dropout = dropout
        self.ingc = GraphConvolution(nfeat, nhid, F.relu)
        self.reslayer = GCFlatResBlock(nhid, nclass, nhid, nreslayer, dropout)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.attention.size(1))
        # self.attention.data.uniform_(-stdv, stdv)
        # print(self.attention)
        pass

    def forward(self, input, adj):
        x = self.ingc(input, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.reslayer(x, adj)
        # x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)


