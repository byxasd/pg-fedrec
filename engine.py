from utils import *
from metrics import MetronAtK
import random
import copy
from data import UserItemRatingDataset
from torch.utils.data import DataLoader
from torch.distributions.laplace import Laplace


class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self._metron = MetronAtK(top_k=10)
        # self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))  # tensorboard writer
        # self._writer.add_text('config', str(config), 0)
        self.server_model_param = {}
        self.client_model_params = {}
        # explicit feedback
        # self.crit = torch.nn.MSELoss()
        # implicit feedback
        self.crit = torch.nn.BCELoss()
        self.top_k = 10

#生成迭代器
    def instance_user_train_loader(self, user_train_data):
        """instance a user's train loader."""
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(user_train_data[0]),
                                        item_tensor=torch.LongTensor(user_train_data[1]),
                                        target_tensor=torch.FloatTensor(user_train_data[2]))
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

    def fed_train_single_batch(self, model_client, batch_data, optimizers, user):
        """train a batch and return an updated model."""
        users, items, ratings = batch_data[0], batch_data[1], batch_data[2]
        ratings = ratings.float()
        reg_item_embedding = copy.deepcopy(self.server_model_param['embedding_item.weight'][user].data)
        optimizer, optimizer_u, optimizer_i = optimizers
        if self.config['use_cuda'] is True:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
            reg_item_embedding = reg_item_embedding.cuda()
        optimizer.zero_grad()
        optimizer_u.zero_grad()
        optimizer_i.zero_grad()
        #计算得到预测的得分
        ratings_pred = model_client(items)
        loss = self.crit(ratings_pred.view(-1), ratings)
        regularization_term = compute_regularization(model_client, reg_item_embedding)
        loss += self.config['reg'] * regularization_term
        loss.backward()
        optimizer.step()
        optimizer_u.step()
        optimizer_i.step()
        return model_client, loss.item()

#对所有client的model的参数进行聚合
    def graph_guide_aggregate(self, round_user_params):
        #通过模型参数的相似性进行图聚合的构建
        user_relation_graph = construct_user_relation_graph_via_item(round_user_params, self.config['num_items'],
                                                            self.config['latent_dim'],
                                                            self.config['similarity_metric'])

        # select the top-k neighborhood for each user.
        topk_user_relation_graph = select_topk_neighboehood(user_relation_graph, self.config['neighborhood_size'],
                                                            self.config['neighborhood_threshold'])
        # 进行model的图聚合
        updated_item_embedding = MP_on_graph(round_user_params, self.config['num_items'], self.config['latent_dim'],
                                             topk_user_relation_graph, self.config['mp_layers'])
        #对全局的item进行了更新
        graph_guide_fed_items = copy.deepcopy(updated_item_embedding)
        return graph_guide_fed_items
    # def graph_guide_aggregate(self, round_user_params):
    #     # --- 聚合前相似度矩阵 ---
    #     user_relation_graph = construct_user_relation_graph_via_item(
    #         round_user_params,
    #         self.config['num_items'],
    #         self.config['latent_dim'],
    #         self.config['similarity_metric']
    #     )
    #
    #     # 如果是距离，转成相似度
    #     if self.config['similarity_metric'] == 'cosine':
    #         sim_before = 1 - user_relation_graph
    #     else:
    #         sim_before = -user_relation_graph
    #
    #     print("聚合前用户相似度矩阵：")
    #     print(sim_before)
    #
    #     # --- 选 top-k 邻居 ---
    #     topk_user_relation_graph = select_topk_neighboehood(
    #         user_relation_graph,
    #         self.config['neighborhood_size'],
    #         self.config['neighborhood_threshold']
    #     )
    #
    #     # --- 图聚合 ---
    #     updated_item_embedding = MP_on_graph(
    #         round_user_params,
    #         self.config['num_items'],
    #         self.config['latent_dim'],
    #         topk_user_relation_graph,
    #         self.config['mp_layers']
    #     )
    #
    #     # --- 聚合后相似度矩阵 ---
    #     # 构造聚合后 embedding
    #     num_users = len(round_user_params)
    #     item_num = self.config['num_items']
    #     latent_dim = self.config['latent_dim']
    #     item_embedding_after = np.zeros((num_users, item_num * latent_dim))
    #     for user in range(num_users):
    #         item_embedding_after[user] = updated_item_embedding[user].numpy().flatten()
    #
    #     adj_after = pairwise_distances(item_embedding_after, metric=self.config['similarity_metric'])
    #     if self.config['similarity_metric'] == 'cosine':
    #         sim_after = 1 - adj_after
    #     else:
    #         sim_after = -adj_after
    #
    #     print("聚合后用户相似度矩阵：")
    #     print(sim_after)
    #
    #     # --- 返回聚合后的 embedding ---
    #     graph_guide_fed_items = copy.deepcopy(updated_item_embedding)
    #     return graph_guide_fed_items, sim_before, sim_after

    #进行平均聚合
    # def aggregate_clients_params(self, round_user_params,participants):
    #     """aggregate a user's train loader."""
    #     updated_item_embedding = aggAvg(round_user_params)
    #     for user in participants:
    #         self.server_model_param['embedding_item.weight'][user] = copy.deepcopy(updated_item_embedding[user])
    #     self.server_model_param['embedding_item.weight']['global'] = copy.deepcopy(updated_item_embedding['global'])

    def aggregate_clients_params(self, round_user_params,participants):
        """aggregate a user's train loader."""
        # 存储所有edge_server要传输到中央的数据
        edge_item_params = {}
        num_users = self.config['num_users']
        num_edges = self.config['edge_servers_num']

        # 计算每个边缘服务器应处理的客户端数量
        clients_per_edge = num_users // num_edges

        for i in range(num_edges):
            # 计算当前边缘服务器的客户端范围
            start_idx = i * clients_per_edge
            end_idx = start_idx + clients_per_edge
            # 确保索引不越界
            end_idx = min(end_idx, num_users)

            # 存储一个edgeserver接收到的数据
            edge_item_embedding = {}
            for client_idx in range(start_idx, end_idx):
                edge_item_embedding[client_idx]={}
                # 使用连续索引，避免键冲突
                edge_item_embedding[client_idx]['embedding_item.weight'] =round_user_params[client_idx]['embedding_item.weight']

            # 对edge服务器存储的数据进行聚合之后得到的edge数据
            if edge_item_embedding:  # 确保不为空
                edge_item_params[i]={}
                edge_item_params[i]['embedding_item.weight'] = aggAvg(edge_item_embedding)['global']

        # 对全局的所有边缘服务器进行聚合，得到最终的模型
        if edge_item_params:  # 确保不为空
            updated_item_embedding = self.graph_guide_aggregate(edge_item_params)
            for user in participants:
                self.server_model_param['embedding_item.weight'][user] = copy.deepcopy(updated_item_embedding['global'])
        self.server_model_param['embedding_item.weight']['global'] = copy.deepcopy(updated_item_embedding['global'])

    def fed_train_a_round(self, all_train_data, round_id):
        """train a round."""
        # 随机选择参加训练的client
        num_participants = int(self.config['num_users'] * self.config['clients_sample_ratio'])
        participants = random.sample(range(self.config['num_users']), num_participants)
        # store users' model parameters of current round.
        round_participant_params = {}

        # 在第一次进行训练时，对所有的client进行参数的初始化
        if round_id == 0:
            self.server_model_param['embedding_item.weight'] = {}
            '''server_model_param存放了所有user从server中加载得到的模型'''
            for user in participants:
                self.server_model_param['embedding_item.weight'][user] = copy.deepcopy(self.model.state_dict()['embedding_item.weight'].data.cpu())
            self.server_model_param['embedding_item.weight']['global'] = copy.deepcopy(self.model.state_dict()['embedding_item.weight'].data.cpu())
        #进行本地的训练
        for user in participants:
            #从全局model加载模型
            model_client = copy.deepcopy(self.model)
            '''model_client是该用户获得的模型参数'''
            # for the first round, client models copy initialized parameters directly.
            # for other rounds, client models receive updated user embedding and aggregated item embedding from server
            # and use local updated mlp parameters from last round.
            if round_id != 0:
                # for participated users, load local updated parameters.
                user_param_dict = copy.deepcopy(self.model.state_dict())
                if user in self.client_model_params.keys():
                    for key in self.client_model_params[user].keys():
                        user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).cuda()
                user_param_dict['embedding_item.weight'] = copy.deepcopy(self.server_model_param['embedding_item.weight']['global'].data).cuda()
                model_client.load_state_dict(user_param_dict)
            # Defining optimizers
            #对mlp线性变化的参数的更新
            optimizer = torch.optim.SGD(
                [{"params": model_client.fc_layers.parameters()}, {"params": model_client.affine_output.parameters()}],
                lr=self.config['lr'])  # MLP optimizer
            # 对本地user的嵌入矩阵进行训练
            optimizer_u = torch.optim.SGD(model_client.embedding_user.parameters(),
                                          lr=self.config['lr'] / self.config['clients_sample_ratio'] * self.config[
                                              'lr_eta'] - self.config['lr'])  # User optimizer
            # 对item进行训练
            optimizer_i = torch.optim.SGD(model_client.embedding_item.parameters(),
                                          lr=self.config['lr'] * self.config['num_items'] * self.config['lr_eta'] -
                                             self.config['lr'])

            optimizers = [optimizer, optimizer_u, optimizer_i]
            # 提取user的训练数据
            user_train_data = [all_train_data[0][user], all_train_data[1][user], all_train_data[2][user]]
            user_dataloader = self.instance_user_train_loader(user_train_data)
            model_client.train()
            # 进行训练
            for epoch in range(self.config['local_epoch']):
                for batch_id, batch in enumerate(user_dataloader):
                    assert isinstance(batch[0], torch.LongTensor)
                    #对传入的model_client进行训练，得到更新之后的model
                    model_client, loss = self.fed_train_single_batch(model_client, batch, optimizers, user)
            # print('[User {}]'.format(user))
            # obtain client model parameters.
            client_param = model_client.state_dict()
            # 存放client训练得到的模型
            '''client_model_params[user]存放了client训练之后得到的完整模型'''
            self.client_model_params[user] = copy.deepcopy(client_param)
            for key in self.client_model_params[user].keys():
                self.client_model_params[user][key] = self.client_model_params[user][key].data.cpu()
            # round_participant_params[user] = copy.deepcopy(self.client_model_params[user])
            # del round_participant_params[user]['embedding_user.weight']
            round_participant_params[user] = {}
            round_participant_params[user]['embedding_item.weight'] = copy.deepcopy(self.client_model_params[user]['embedding_item.weight'])
            #round_participant_params[user]['embedding_item.weight'] += Laplace(0, self.config['dp']).expand(round_participant_params[user]['embedding_item.weight'].shape).sample()
        # 进行全局的model的聚合
        self.aggregate_clients_params(round_participant_params,participants)
        return participants

    def fed_evaluate(self, evaluate_data):
        # evaluate all client models' performance using testing data.
        test_users, test_items = evaluate_data[0], evaluate_data[1]    #正采样
        negative_users, negative_items = evaluate_data[2], evaluate_data[3]   #负采样
        # ratings for computing loss.
        temp = [0] * 100
        temp[0] = 1
        ratings = torch.FloatTensor(temp)
        '''放入cuda中进行计算'''
        if self.config['use_cuda'] is True:
            test_users = test_users.cuda()
            test_items = test_items.cuda()
            negative_users = negative_users.cuda()
            negative_items = negative_items.cuda()
            ratings = ratings.cuda()
        # store all users' test item prediction score.
        test_scores = None
        # store all users' negative items prediction scores.
        negative_scores = None
        all_loss = {}
        for user in range(self.config['num_users']):
            #加载全局的模型
            user_model = copy.deepcopy(self.model)
            user_param_dict = copy.deepcopy(self.model.state_dict())
            if user in self.client_model_params.keys():
                #加载该user的数据
                for key in self.client_model_params[user].keys():
                    user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).cuda()
            # user_param_dict['embedding_item.weight'] = copy.deepcopy(
            #     self.server_model_param['embedding_item.weight']['global'].data).cuda()
            user_model.load_state_dict(user_param_dict)
            user_model.eval()
            with torch.no_grad():
                # obtain user's positive test information.
                test_user = test_users[user: user + 1]
                test_item = test_items[user: user + 1]
                # obtain user's negative test information.
                negative_user = negative_users[user * 99: (user + 1) * 99]
                negative_item = negative_items[user * 99: (user + 1) * 99]
                # perform model prediction.
                #得到rating
                test_score = user_model(test_item)
                negative_score = user_model(negative_item)
                if user == 0:
                    test_scores = test_score
                    negative_scores = negative_score
                else:
                    test_scores = torch.cat((test_scores, test_score))
                    negative_scores = torch.cat((negative_scores, negative_score))
                #计算得到预测得分
                ratings_pred = torch.cat((test_score, negative_score))
                loss = self.crit(ratings_pred.view(-1), ratings)
            all_loss[user] = loss.item()
        if self.config['use_cuda'] is True:
            test_users = test_users.cpu()
            test_items = test_items.cpu()
            test_scores = test_scores.cpu()
            negative_users = negative_users.cpu()
            negative_items = negative_items.cpu()
            negative_scores = negative_scores.cpu()
        self._metron.subjects = [test_users.data.view(-1).tolist(),
                                 test_items.data.view(-1).tolist(),
                                 test_scores.data.view(-1).tolist(),
                                 negative_users.data.view(-1).tolist(),
                                 negative_items.data.view(-1).tolist(),
                                 negative_scores.data.view(-1).tolist()]
        hit_ratio, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg()
        return hit_ratio, ndcg, all_loss

    def save(self, alias, epoch_id, hit_ratio, ndcg):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = self.config['model_dir'].format(alias, epoch_id, hit_ratio, ndcg)
        save_checkpoint(self.model, model_dir)
