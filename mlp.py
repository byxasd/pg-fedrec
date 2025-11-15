import torch
from engine import Engine
from utils import use_cuda, resume_checkpoint


#整体的模型参数
class MLP(torch.nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        #本地的user的嵌入矩阵
        self.embedding_user = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.latent_dim)
        #item的嵌入矩阵
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        #线性层
        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
        #最后的变化层
        self.affine_output = torch.nn.Linear(in_features=config['layers'][-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, item_indices):
        user_embedding = self.embedding_user(torch.LongTensor([0 for i in range(len(item_indices))]).cuda())
        #传入item的索引，得到对应的嵌入矩阵
        item_embedding = self.embedding_item(item_indices)
        #连接嵌入矩阵
        vector = torch.cat([user_embedding, item_embedding], dim=-1)
        #进行线性变化
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)
            # vector = torch.nn.BatchNorm1d()(vector)
            # vector = torch.nn.Dropout(p=0.5)(vector)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass


class MLPEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        self.model = MLP(config)
        #选择cuda：0
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        #初始化联邦训练的引擎
        super(MLPEngine, self).__init__(config)
        print(self.model)
