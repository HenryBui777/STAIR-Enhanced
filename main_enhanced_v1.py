from typing import Dict, Tuple, Optional
import torch, os, math
import torch.nn as nn
import torch.nn.functional as F
import freerec

from optimizers.Adam import AdamSEvo
from optimizers.AdamW import AdamWSEvo
from optimizers.utils import Smoother
from models.enhanced_projector_v2 import DeRedundantGatedProjector, prepare_item_embeddings

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--embedding-dim",  type=int,   default=64)
cfg.add_argument("--num-layers",     type=int,   default=3, help="the number of layers for FSC/BSC")
cfg.add_argument("--mfiles",         type=str,   default="textual_modality.pkl,visual_modality.pkl", help="the files saving modality")
cfg.add_argument("--num-neighbors",  type=str,   default='5-1', help="for kNN graph")
cfg.add_argument("--gamma",          type=float, default=0.2)

# Module 1 (De-redundant Gated Projector) hyperparams
cfg.add_argument("--ema-decay",      type=float, default=0.99, help="EMA decay for covariance tracking in DeRedundantGatedProjector")
cfg.add_argument("--null-rank",      type=int,   default=16,   help="Number of top PCs removed in null-space projection")

cfg.set_defaults(
    description="EnhancedSTAIR-v1",
    root="../../data",
    dataset='Amazon2014Baby_550_MMRec',
    epochs=500,
    batch_size=1024,
    optimizer='adamwsevo',
    lr=1e-3,
    weight_decay=0.1,
    seed=1,
    monitors=["Recall@10", "Recall@20", "NDCG@10", "NDCG@20"],
    which4best="NDCG@20",
)
cfg.compile()

cfg.mfiles        = cfg.mfiles.split(',')
cfg.num_neighbors = list(map(int, cfg.num_neighbors.split('-')))
cfg.beta3 = (
    0.1 + 0.9 * (torch.arange(cfg.embedding_dim) / cfg.embedding_dim).pow(cfg.gamma)
).to(cfg.device)


class EnhancedSTAIR(freerec.models.GenRecArch):
    """
    STAIR with Module 1 (DeRedundantGatedProjector) replacing static SVD Whitening.
    FSC / BSC / BPR core is UNCHANGED from the original paper.
    """

    def __init__(self, dataset: freerec.data.datasets.RecDataSet) -> None:
        super().__init__(dataset)
        self.num_layers = cfg.num_layers

        self.User.add_module('embeddings', nn.Embedding(self.User.count, cfg.embedding_dim))
        self.Item.add_module('embeddings', nn.Embedding(self.Item.count, cfg.embedding_dim))

        self.register_buffer(
            'Adj',
            self.dataset.train().to_normalized_adj(normalization='sym')
        )

        # Module 1: learnable De-redundant Gated Projector (replaces static whitening)
        self.projector = DeRedundantGatedProjector(
            d_text=384,
            d_visual=4096,
            d_hidden=cfg.embedding_dim,
            ema_decay=cfg.ema_decay,
            null_rank=cfg.null_rank,
        )

        self.reset_parameters()
        self.prepare(dataset.path)
        self.criterion = freerec.criterions.BPRLoss(reduction='mean')

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=1.e-4)

    def marked_params(self):
        return [
            {'params': self.User.parameters(),      'smoother': None},
            {'params': self.Item.parameters(),      'smoother': Smoother(self.mAdj, beta=cfg.beta3, L=cfg.num_layers, aggr='neumann')},
            {'params': self.projector.parameters(), 'smoother': None},
        ]

    def get_knn_graph(self, features: torch.Tensor, k: int = 5):
        features = F.normalize(features, dim=-1)
        sim = features @ features.t()
        sim.fill_diagonal_(-10.)
        edge_index, _ = freerec.graph.get_knn_graph(sim, k, symmetric=False)
        return edge_index

    def prepare(self, path: str):
        from freerec.utils import import_pickle

        mfeats_raw = [
            import_pickle(os.path.join(path, mfile))
            for mfile in cfg.mfiles
        ]

        edge_index = torch.cat(
            [self.get_knn_graph(feats, k) for feats, k in zip(mfeats_raw, cfg.num_neighbors)],
            dim=1
        )
        edge_weight = torch.ones_like(edge_index[0], dtype=torch.float)
        edge_index, edge_weight = freerec.graph.coalesce(edge_index, edge_weight, reduce='sum')
        edge_index, edge_weight = freerec.graph.to_undirected(edge_index, edge_weight, reduce='max')
        edge_index, edge_weight = freerec.graph.to_normalized(edge_index, edge_weight, normalization='sym')
        mAdj = torch.sparse_coo_tensor(
            edge_index, edge_weight,
            size=(self.Item.count, self.Item.count)
        )
        self.register_buffer('mAdj', mAdj.to_sparse_csr())

        # Module 1 Plug-in: bypass whitening() + weighted sum
        text_feat, visual_feat = mfeats_raw[0], mfeats_raw[1]
        mfeats = prepare_item_embeddings(
            self.projector, text_feat, visual_feat,
            n_items=self.Item.count, embedding_dim=cfg.embedding_dim
        )
        self.Item.embeddings.weight.data.copy_(mfeats)

        # User init via R @ mfeats (identical to STAIR)
        edge_index_ui = self.dataset.train().to_bigraph(edge_type='u2i')['u2i'].edge_index
        edge_index_ui, edge_weight_ui = freerec.graph.to_normalized(edge_index_ui, normalization='left')
        R = torch.sparse_coo_tensor(
            edge_index_ui, edge_weight_ui,
            size=(self.User.count, self.Item.count)
        ).to_sparse_csr()
        self.User.embeddings.weight.data.copy_(R @ mfeats)

    def sure_trainpipe(self, batch_size: int):
        return self.dataset.train().shuffled_pairs_source(
        ).gen_train_sampling_neg_(num_negatives=1).batch_(batch_size).tensor_()

    def encode(self) -> Tuple[torch.Tensor, torch.Tensor]:
        allEmbds = torch.cat((self.User.embeddings.weight, self.Item.embeddings.weight), dim=0)
        features = allEmbds
        smoothed = allEmbds
        beta = 1 - cfg.beta3
        norm_correction = 1 - beta ** (self.num_layers + 1)
        for _ in range(self.num_layers):
            features = self.Adj @ features * beta
            smoothed = smoothed + features
        avgEmbds = smoothed.mul(1 - beta).div(norm_correction)
        userEmbds, itemEmbds = torch.split(avgEmbds, (self.User.count, self.Item.count))
        return userEmbds, itemEmbds

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds, itemEmbds = self.encode()
        users, positives, negatives = data[self.User], data[self.Item], data[self.INeg]
        userEmbds = userEmbds[users]
        iposEmbds = itemEmbds[positives]
        inegEmbds = itemEmbds[negatives]
        return self.criterion(
            torch.einsum('BKD,BKD->BK', userEmbds, iposEmbds),
            torch.einsum('BKD,BKD->BK', userEmbds, inegEmbds)
        )

    def reset_ranking_buffers(self):
        userEmbds, itemEmbds = self.encode()
        self.ranking_buffer = dict()
        self.ranking_buffer[self.User] = userEmbds.detach().clone()
        self.ranking_buffer[self.Item] = itemEmbds.detach().clone()

    def recommend_from_full(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds = self.ranking_buffer[self.User][data[self.User]]
        itemEmbds = self.ranking_buffer[self.Item]
        return torch.einsum('BKD,ND->BN', userEmbds, itemEmbds)

    def recommend_from_pool(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds = self.ranking_buffer[self.User][data[self.User]]
        itemEmbds = self.ranking_buffer[self.Item][data[self.IUnseen]]
        return torch.einsum('BKD,BKD->BK', userEmbds, itemEmbds)


class CoachForEnhancedSTAIR(freerec.launcher.Coach):

    def set_optimizer(self):
        self.optimizer = AdamWSEvo(
            self.model.marked_params(), lr=self.cfg.lr,
            betas=(self.cfg.beta1, self.cfg.beta2),
            weight_decay=self.cfg.weight_decay
        )

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            data = self.dict_to_device(data)
            loss = self.model(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.monitor(
                loss.item(), n=len(data[self.User]),
                reduction='mean', mode='train', pool=['LOSS']
            )


def main():
    # Robust dataset loading: freerec.data.datasets contains a submodule named 'tiktok',
    # so getattr(...) returns a module rather than a class when cfg.dataset == 'tiktok'.
    ds_cls = getattr(freerec.data.datasets, cfg.dataset, None)
    if isinstance(ds_cls, type):
        try:
            dataset = ds_cls(root=cfg.root)
        except Exception:
            dataset = freerec.data.datasets.RecDataSet(
                cfg.root, cfg.dataset, tasktag=getattr(cfg, "tasktag", None)
            )
    else:
        dataset = freerec.data.datasets.RecDataSet(
            cfg.root, cfg.dataset, tasktag=getattr(cfg, "tasktag", None)
        )

    model = EnhancedSTAIR(dataset)
    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe  = model.sure_testpipe(cfg.ranking)

    coach = CoachForEnhancedSTAIR(
        dataset=dataset, trainpipe=trainpipe,
        validpipe=validpipe, testpipe=testpipe,
        model=model, cfg=cfg
    )
    coach.fit()


if __name__ == '__main__':
    main()
