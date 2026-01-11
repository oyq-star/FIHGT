import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn import HGTConv, RGCNConv
from layer import RGTLayer, SimpleHGNLayer
import torch.nn.functional as F
from itertools import product
import torch_geometric
import networkx as nx
import scipy
from scipy.io import savemat

class BotRGCN(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=3, relation_num=2, dropout=0.3):
        super(BotRGCN, self).__init__()
        self.dropout = dropout

        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(768, int(hidden_dimension*3/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(10, int(hidden_dimension/8)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(10, int(hidden_dimension/8)),
            nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LeakyReLU()
        )

        self.rgcn = RGCNConv(hidden_dimension, hidden_dimension, num_relations=relation_num)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dimension, out_dim)

    def forward(self, feature, edge_index, edge_type):
        t = self.linear_relu_tweet(feature[:, -768:].to(torch.float32))
        n = self.linear_relu_num_prop(feature[:, [4,6,7,8,10,11,12,13,14,15]].to(torch.float32))
        b = self.linear_relu_cat_prop(feature[:, [1,2,3,5,9,16,17,18,19,20]].to(torch.float32))
        x = torch.cat((t, n, b), dim=1)
        x = self.linear_relu_input(x)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        # x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        user_features = x
        return x,user_features



class RGCN(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=3, relation_num=2, dropout=0.3):
        super(RGCN, self).__init__()
        self.dropout = dropout
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.rgcn1 = RGCNConv(hidden_dimension, hidden_dimension, num_relations=relation_num)
        self.rgcn2 = RGCNConv(hidden_dimension, hidden_dimension, num_relations=relation_num)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dimension, out_dim)

    def forward(self, feature, edge_index, edge_type):
        x = self.linear_relu_input(feature.to(torch.float32))
        x = self.rgcn1(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn1(x, edge_index, edge_type)
        user_features=x
        # x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x,user_features



class GAT(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=3, relation_num=2, dropout=0.3):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.LeakyReLU()
        )

        self.gat1 = GATConv(hidden_dimension, int(hidden_dimension / 8), heads=8)
        self.gat2 = GATConv(hidden_dimension, hidden_dimension)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dimension, out_dim)

    def forward(self, feature, edge_index, edge_type):
        x = self.linear_relu_input(feature.to(torch.float32))
        x = self.gat1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        user_features = x
        # x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x,user_features



class GCN(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=3, relation_num=2, dropout=0.3):
        super(GCN, self).__init__()
        self.dropout = dropout

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.LeakyReLU()
        )

        self.gcn1 = GCNConv(hidden_dimension, hidden_dimension)
        self.gcn2 = GCNConv(hidden_dimension, hidden_dimension)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dimension, out_dim)

    def forward(self, feature, edge_index, edge_type):
        x = self.linear_relu_input(feature.to(torch.float32))
        x = self.gcn1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, edge_index)
        user_features = x
        # x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x,user_features



class SAGE(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=3, relation_num=2, dropout=0.3):
        super(SAGE, self).__init__()
        self.dropout = dropout

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.LeakyReLU()
        )

        self.sage1 = SAGEConv(hidden_dimension, hidden_dimension)
        self.sage2 = SAGEConv(hidden_dimension, hidden_dimension)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dimension, out_dim)

    def forward(self, feature, edge_index, edge_type):
        x = self.linear_relu_input(feature.to(torch.float32))
        x = self.sage1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.sage2(x, edge_index)
        user_features = x
        # x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x,user_features



class HGT(nn.Module):
    def __init__(self, args, relation_list):
        super(HGT, self).__init__()

        self.relation_list = list(relation_list)
        self.linear1 = nn.Linear(args.features_num, args.hidden_dimension)

        self.HGT_layer1 = HGTConv(in_channels=args.hidden_dimension, out_channels=args.hidden_dimension,
                                  metadata=(['user'], self.relation_list))
        self.HGT_layer2 = HGTConv(in_channels=args.hidden_dimension, out_channels=args.linear_channels,
                                  metadata=(['user'], self.relation_list))
        self.out1 = torch.nn.Linear(args.linear_channels, args.out_channel)
        self.out2 = torch.nn.Linear(args.out_channel, args.out_dim)

        self.drop = nn.Dropout(args.dropout)
        self.CELoss = nn.CrossEntropyLoss()
        self.ReLU = nn.LeakyReLU()

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, features, edge_index_dict):

        user_features = self.drop(self.ReLU(self.linear1(features)))
        x_dict = {"user": user_features}
        x_dict = self.HGT_layer1(x_dict, edge_index_dict)
        x_dict = self.HGT_layer1(x_dict, edge_index_dict)
        user_features = self.ReLU(self.out1(x_dict["user"]))
        x = self.out2(user_features)

        return x


class FIHGT(nn.Module):
    def __init__(self, args,
                 # feature_map,
                 model_id="FiGNN",
                 gpu=-1,
                 learning_rate=1e-3,
                 #num_fields=5301,
                 #num_fields=11826,
                 num_fields=10199,
                 #num_fields=5301,
                 embedding_dim=64,
                 gnn_layers=2,
                 use_residual=True,
                 use_gru=True,
                 reuse_graph_layer=False,
                 embedding_regularizer=1e-8,
                 net_regularizer=0,
                 **kwargs):
        super(FIHGT, self).__init__(
           #  # feature_map,
           # # model_id=model_id,
           #  #gpu=gpu,
           #  embedding_regularizer=embedding_regularizer,
           #  net_regularizer=net_regularizer,
           #  **kwargs
        )



        self.linear1 = nn.Linear(args.features_num, args.hidden_dimension)
        self.RGT_layer1 = RGTLayer(num_edge_type=len(args.relation_select), in_channel=args.hidden_dimension, out_channel=args.hidden_dimension,
                                   trans_heads=args.trans_head, semantic_head=args.semantic_head, dropout=args.dropout)
        self.RGT_layer2 = RGTLayer(num_edge_type=len(args.relation_select), in_channel=args.hidden_dimension, out_channel=args.hidden_dimension, trans_heads=args.trans_head, semantic_head=args.semantic_head, dropout=args.dropout)

        self.out1 = torch.nn.Linear(args.hidden_dimension, args.out_channel)
        self.out2 = torch.nn.Linear(args.out_channel, args.out_dim)

        self.drop = nn.Dropout(args.dropout)
        self.CELoss = nn.CrossEntropyLoss()
        self.ReLU = nn.LeakyReLU()

        self.init_weight()




        #num_fields = gnn_layers,
        self.fignn = FiGNN_Layer(num_fields,
                                 embedding_dim,
                                 gnn_layers=gnn_layers,
                                 reuse_graph_layer=reuse_graph_layer,
                                 use_gru=use_gru,
                                 use_residual=use_residual)
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, features, edge_index, edge_type):



        adj = torch_geometric.utils.to_scipy_sparse_matrix(edge_index)

        #h_out = self.fignn(features, adj)
        #features=h_out
        user_features = self.drop(self.ReLU(self.linear1(features)))

        #user_features = self.ReLU(self.RGT_layer1(user_features, edge_index, edge_type))
        user_features = self.ReLU(self.RGT_layer1(user_features, edge_index, edge_type))

        #h_out = self.fignn(user_features, adj)
        # user_features=h_out
        user_features = self.drop(self.ReLU(self.out1(user_features)))
        h_out = self.fignn(user_features, adj)
        user_features=h_out
        #savemat('a.mat', user_features)
        x = self.out2(user_features)

        return x,user_features



class SHGN(nn.Module):
    def __init__(self, args):
        super(SHGN, self).__init__()

        self.linear1 = nn.Linear(args.features_num, args.hidden_dimension)
        self.HGN_layer1 = SimpleHGNLayer(num_edge_type=args.num_edge_type, in_channels=args.hidden_dimension,
                                         out_channels=args.hidden_dimension, rel_dim=args.rel_dim, beta=args.beta)
        self.HGN_layer2 = SimpleHGNLayer(num_edge_type=args.num_edge_type, in_channels=args.hidden_dimension,
                                         out_channels=args.linear_channels, rel_dim=args.rel_dim, beta=args.beta,
                                         final_layer=True)

        self.out1 = torch.nn.Linear(args.linear_channels, args.out_channel)
        self.out2 = torch.nn.Linear(args.out_channel, args.out_dim)

        self.drop = nn.Dropout(args.dropout)
        self.ReLU = nn.LeakyReLU()
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, feature, edge_index, edge_type):

        user_features = self.drop(self.ReLU(self.linear1(feature)))
        user_features, alpha = self.HGN_layer1(user_features, edge_index, edge_type)
        user_features, _ = self.HGN_layer1(user_features, edge_index, edge_type, alpha)
        user_features = self.drop(self.ReLU(self.out1(user_features)))
        x = self.out2(user_features)
        return x

class FiGNN_Layer(nn.Module):
    def __init__(self,
                 num_fields,
                 embedding_dim,
                 gnn_layers=3,
                 reuse_graph_layer=False,
                 use_gru=True,
                 use_residual=True):
        super(FiGNN_Layer, self).__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.gnn_layers = gnn_layers
        self.use_residual = use_residual
        self.reuse_graph_layer = reuse_graph_layer
        if reuse_graph_layer:
            self.gnn = GraphLayer(num_fields, embedding_dim)
        else:
            self.gnn = nn.ModuleList([GraphLayer(num_fields, embedding_dim)
                                      for _ in range(gnn_layers)])
        self.gru = nn.GRUCell(embedding_dim, embedding_dim) if use_gru else None
        #self.src_nodes, self.dst_nodes = zip(*list(product(range(num_fields), repeat=2)))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.W_attn = nn.Linear(embedding_dim * 2, 1, bias=False)

    # def build_graph_with_attention(self, feature_emb):
    #     # src_emb = feature_emb[:, self.src_nodes, :]
    #     # dst_emb = feature_emb[:, self.dst_nodes, :]
    #     src_emb = feature_emb[:, self.src_nodes]
    #     dst_emb = feature_emb[:, self.dst_nodes]
    #     concat_emb = torch.cat([src_emb, dst_emb], dim=-1)
    #     alpha = self.leaky_relu(self.W_attn(concat_emb))
    #     alpha = alpha.view(-1, self.num_fields, self.num_fields)
    #     mask = torch.eye(self.num_fields).to(feature_emb.device)
    #     alpha = alpha.masked_fill(mask.bool(), float('-inf'))
    #     graph = F.softmax(alpha, dim=-1)  # batch x field x field without self-loops
    #     return graph

    def forward(self, feature_emb,adj):
        #g = self.build_graph_with_attention(feature_emb)
        h = feature_emb
        g=torch.FloatTensor(adj.todense())

        for i in range(self.gnn_layers):
            if self.reuse_graph_layer:
                a = self.gnn(g, h)
            else:
                a = self.gnn[i](g, h)
            if self.gru is not None:
                a = a.view(-1, self.embedding_dim)
                h = h.view(-1, self.embedding_dim)
                h = self.gru(a, h)
                #h = h.view(-1, self.num_fields, self.embedding_dim)
            else:
                h = a + h
            #h=torch.reshape(h,(10199,1,128))
            if self.use_residual:
                h += feature_emb
        return h

class GraphLayer(nn.Module):
    def __init__(self, num_fields, embedding_dim):
        super(GraphLayer, self).__init__()
        # self.W_in = torch.nn.Parameter(torch.Tensor(num_fields, embedding_dim, embedding_dim))
        # self.W_out = torch.nn.Parameter(torch.Tensor(num_fields, embedding_dim, embedding_dim))
        self.W_in = torch.nn.Parameter(torch.Tensor(num_fields, embedding_dim)).cuda()
        self.W_out = torch.nn.Parameter(torch.Tensor(num_fields, embedding_dim)).cuda()
        nn.init.xavier_normal_(self.W_in)
        nn.init.xavier_normal_(self.W_out)
        self.bias_p = nn.Parameter(torch.zeros(embedding_dim))

    def add(self,g):
        p = torch.zeros(g.shape[0])
        p = p.unsqueeze(1)
        n = torch.cat((g,p),1)
        p = torch.zeros(g.shape[0]+1)
        p = p.unsqueeze(0)
        n = torch.cat((n, p), 0)
        return n

    def forward(self, g, h):
        h_out = torch.matmul(self.W_out, h.unsqueeze(-1)).squeeze(-1)  # broadcast multiply
        g=self.add(g)
        aggr = torch.matmul(g.cuda(), h_out.unsqueeze(-1)).squeeze(-1)
        a = torch.matmul(aggr,self.W_in) + self.bias_p
        return a
