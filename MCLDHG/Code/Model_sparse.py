import torch as t
from torch import nn
import torch.nn.functional as F
from Params import args
from Utils.Utils import contrastLoss, ce, l2_norm


init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class DHGNN(nn.Module):
    """

    """
    def __init__(self):
        super(DHGNN, self).__init__()

        self.dEmbeds = nn.Parameter(init(t.empty(args.drug, args.latdim)))
        self.mEmbeds = nn.Parameter(init(t.empty(args.microbe, args.latdim)))
        self.disEmbeds = nn.Parameter(init(t.empty(args.disease, args.latdim)))


        self.gcnLayer = GCNLayer()


        self.hgnnLayer = HGNNLayer()


        self.classifierLayer = MLP()
        

        if args.dense:

            self.dHyper = nn.Parameter(init(t.empty(args.latdim, args.hyperNum)))
            self.mHyper = nn.Parameter(init(t.empty(args.latdim, args.hyperNum)))
            self.disHyper = nn.Parameter(init(t.empty(args.latdim, args.hyperNum)))


        self.edgeDropper = SpAdjDropEdge()

    def forward(self, adj, keepRate):
        """

        """

        embeds = t.concat([self.dEmbeds, self.mEmbeds, self.disEmbeds], axis=0)
        embedsLst = [embeds]
        gcnEmbedsLst = [embeds]
        hyperEmbedsLst = [embeds]


        ddHyper = self.dEmbeds * args.mult
        mmHyper = self.mEmbeds * args.mult
        disHyper = self.disEmbeds * args.mult


        if args.dense:   

            ddHyper = self.dEmbeds @ self.dHyper
            mmHyper = self.mEmbeds @ self.mHyper
            disHyper = self.disEmbeds @ self.disHyper


        for i in range(args.gnn_layer):

            gcnEmbeds = self.gcnLayer(self.edgeDropper(adj, keepRate), embedsLst[-1])


            hyperDEmbeds = self.hgnnLayer(ddHyper, embedsLst[-1][:args.drug])  

            hyperMEmbeds = self.hgnnLayer(mmHyper, embedsLst[-1][args.drug:args.drug+args.microbe])  

            hyperDisEmbeds = self.hgnnLayer(disHyper, embedsLst[-1][args.drug+args.microbe:])
            

            hyperEmbeds = t.cat([hyperDEmbeds, hyperMEmbeds, hyperDisEmbeds], dim=0)
    

            gcnEmbedsLst.append(gcnEmbeds)
            hyperEmbedsLst.append(hyperEmbeds)

            embedsLst.append(gcnEmbeds + hyperEmbeds)


        embeds = sum(embedsLst)
        return embeds, gcnEmbedsLst, hyperEmbedsLst

    def calcLosses(self, drugs, microbes, diseases, labels, adj, keepRate):
        """

        """

        embeds, gcnEmbedsLst, hyperEmbedsLst = self.forward(adj, keepRate)


        d_embeds = embeds[:args.drug]
        m_embeds = embeds[args.drug:args.drug + args.microbe]
        dis_embeds = embeds[args.drug + args.microbe:]


        d_embeds = d_embeds[drugs]
        m_embeds = m_embeds[microbes]
        dis_embeds = dis_embeds[diseases]


        pre = self.classifierLayer(d_embeds, m_embeds, dis_embeds)
        ceLoss = ce(pre, labels)


        sslLoss = 0
        for i in range(1, args.gnn_layer + 1, 1):

            embeds1 = gcnEmbedsLst[i].detach()

            embeds2 = hyperEmbedsLst[i]
            

            sslLoss += contrastLoss(embeds1[:args.drug], embeds2[:args.drug],
                                    t.unique(drugs), args.temp)

            sslLoss += contrastLoss(embeds1[args.drug:args.drug + args.microbe],
                                    embeds2[args.drug:args.drug + args.microbe],
                                    t.unique(microbes), args.temp)

            sslLoss += contrastLoss(embeds1[args.drug + args.microbe:],
                                    embeds2[args.drug + args.microbe:],
                                    t.unique(diseases), args.temp)


        return ceLoss, sslLoss

    def predict(self, adj, drugs, microbes, diseases):
        """

        """

        embeds, _, _ = self.forward(adj, 1.0)
        

        dEmbeds = embeds[:args.drug]
        mEmbeds = embeds[args.drug:args.drug + args.microbe]
        disEmbeds = embeds[args.drug + args.microbe:]


        dEmbeds = dEmbeds[drugs]
        mEmbeds = mEmbeds[microbes]
        disEmbeds = disEmbeds[diseases]


        pre = self.classifierLayer(dEmbeds, mEmbeds, disEmbeds)
        return pre



class GCNLayer(nn.Module):
    """

    """
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        """

        """

        return l2_norm(t.spmm(adj, embeds))



class HGNNLayer(nn.Module):
    """

    """
    def __init__(self):
        super(HGNNLayer, self).__init__()

    def forward(self, adj, embeds):
        """

        """

        lat = adj.T @ embeds

        ret = adj @ lat

        return l2_norm(ret)



class SpAdjDropEdge(nn.Module):
    """

    """
    def __init__(self):
        super(SpAdjDropEdge, self).__init__()

    def forward(self, adj, keepRate):
        """

        """

        if keepRate == 1.0:
            return adj
            

        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        

        mask = ((t.rand(edgeNum) + keepRate).floor()).type(t.bool)

        newVals = vals[mask] / keepRate

        newIdxs = idxs[:, mask]
        

        return t.sparse.FloatTensor(newIdxs, newVals, adj.shape)


class MLP(nn.Module):
    """

    """
    def __init__(self):
        super(MLP, self).__init__()

        self.lin1 = nn.Linear(args.latdim * 3, 128)

        self.lin2 = nn.Linear(128, 2)

    def forward(self, dEmbeds, mEmbeds, disEmbeds):
        """

        """

        embeds = t.concat((dEmbeds, mEmbeds, disEmbeds), 1)

        embeds = F.relu(self.lin1(embeds))

        embeds = F.dropout(embeds, p=0.4, training=self.training)

        ret = self.lin2(embeds)
        return ret
