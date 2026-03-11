import torch as t
import torch.nn.functional as F
import torch
import numpy as np

def calcRegLoss(model):
    """
    Calculate the regularization loss by summing the L2 norm of model parameters.

    Parameters:
    model (torch.nn.Module): The neural network model.

    Returns:
    ret (float): The regularization loss.
    """
    ret = 0
    for W in model.parameters():
        ret += W.norm(2).square()
    return ret


def contrastLoss(embeds1, embeds2, nodes, temp):
    """
    Calculate the contrastive loss for embeddings.

    Parameters:
    embeds1 (torch.Tensor): The first set of embeddings.
    embeds2 (torch.Tensor): The second set of embeddings.
    nodes (torch.Tensor): Indices of nodes used for contrastive loss.
    temp (float): Temperature parameter for the contrastive loss.

    Returns:
    loss (torch.Tensor): The contrastive loss.
    """

    embeds1 = F.normalize(embeds1 + 1e-8, p=2)
    embeds2 = F.normalize(embeds2 + 1e-8, p=2)

    pckEmbeds1 = embeds1[nodes]
    pckEmbeds2 = embeds2[nodes]

    nume = t.exp(t.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)

    deno = t.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1) + 1e-8

    return -t.log(nume / deno).mean()


def ce(pred, target):
    """
    Calculate the cross-entropy loss between predicted and target values.

    Parameters:
    pred (torch.Tensor): Predicted values.
    target (torch.Tensor): Target values.

    Returns:
    loss (torch.Tensor): The cross-entropy loss.
    """
    return F.cross_entropy(pred, target)


def l2_norm(x):
    """
    Calculate L2 normalization of a tensor.

    Parameters:
    x (torch.Tensor): The input tensor.

    Returns:
    normalized_x (torch.Tensor): The L2 normalized tensor.
    """
    # epsilon = t.FloatTensor([1e-12]).cuda()
    epsilon = t.FloatTensor([1e-12])
    # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
    return x / (t.max(t.norm(x, dim=1, keepdim=True), epsilon))

def hit_ndcg_value(pred_val, val_data, top):


    loader_val = torch.utils.data.DataLoader(dataset=pred_val, batch_size=30, shuffle=False)
    hits = 0
    ndcg_val = 0
    for step, batch_val in enumerate(loader_val):
        metrix = Metrics(step, val_data, pred_val, batch_size=30, top=top)
        hit, ndcg = metrix.hits_ndcg()
        hits = hits + hit
        ndcg_val = ndcg_val + ndcg
    hits = hits / int((len(val_data)) / 30)
    ndcg = ndcg_val / int((len(val_data)) / 30)
    return hits, ndcg

class Metrics():
    def __init__(self, step, test_data, predict_1, batch_size, top):
        """
        Args:
            step:
            test_data:
            predict_1:
            batch_size:
            top: top-k
        """
        self.step = step
        self.test_data = test_data
        self.predict_1 = predict_1
        self.batch_size = batch_size
        self.top = top
        

        self.pair = []
        self.val_top = []
        self.hit = 0
        self.dcgsum = 0
        self.idcgsum = 0
        self.ndcg = 0

    def hits_ndcg(self):
        """

        """
        # print('test_data:',self.test_data)
        # print('predict_1:',self.predict_1)

        for i in range(self.step * self.batch_size, (self.step + 1) * self.batch_size):
            if i >= len(self.test_data):
                break
            g = []

            pred_score = self.predict_1[i][1]
            g.extend([
                self.test_data[i],
                pred_score
            ])
            self.pair.append(g)


        np.random.seed(1)
        np.random.shuffle(self.pair)


        pre_val = sorted(self.pair, key=lambda item: item[1], reverse=True)

        self.val_top = pre_val[0: self.top]


        for i in range(len(self.val_top)):
            if self.val_top[i][0] == 1:
                self.hit = self.hit + 1
                self.dcgsum = (2 ** self.val_top[i][0] - 1) / np.log2(i + 2)
                break


        ideal_list = sorted(self.val_top, key=lambda item: item[0], reverse=True)
        for i in range(len(ideal_list)):
            if ideal_list[i][0] == 1:
                self.idcgsum = (2 ** ideal_list[i][0] - 1) / np.log2(i + 2)
                break


        self.ndcg = self.dcgsum / self.idcgsum if self.idcgsum != 0 else 0

        return self.hit, self.ndcg
    
def get_metrics(real_score, predict_score):

    predict_score = predict_score[:, 1]
    

    real_score = real_score.flatten()
    

    sorted_predict_score = np.array(sorted(list(set(predict_score.flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    

    thresholds = sorted_predict_score[
        (np.array([sorted_predict_score_num]) * np.arange(1, 1000) / np.array([1000])).astype(int)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]


    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1


    real_score_matrix = np.mat(real_score)
    predict_score_matrix = np.mat(predict_score_matrix)
    

    TP = predict_score_matrix * real_score_matrix.T
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score_matrix.sum() - TP
    TN = real_score_matrix.shape[1] - TP - FP - FN


    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    

    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])


    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack((recall_list, -precision_list)).tolist())).T
    PR_dot_matrix[1, :] = -PR_dot_matrix[1, :]
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])


    f1_score_list = 2 * TP / (real_score_matrix.shape[1] + TP - TN)
    accuracy_list = (TP + TN) / real_score_matrix.shape[1]
    specificity_list = TN / (TN + FP)


    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index, 0]
    accuracy = accuracy_list[max_index, 0]
    specificity = specificity_list[max_index, 0]
    recall = recall_list[max_index, 0]
    precision = precision_list[max_index, 0]
    
    return [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision]  