import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model_sparse import DHGNN
from DataHandler import DataHandler
from Utils.Utils import *
import os
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import numpy as np
import random
import wandb
import torch
import time


# Function to set random seed for reproducibility
def set_seed(seed):
    """

    """
    print("set_seed: ", seed)

    random.seed(seed)

    np.random.seed(seed)

    t.manual_seed(seed)

    if t.cuda.is_available():
        t.cuda.manual_seed(seed)

        t.backends.cudnn.benchmark = False

        t.backends.cudnn.deterministic = True


# Define the Coach class for model training and evaluation
class Coach:
    def __init__(self, handler):
        self.handler = handler

        print('drug:', args.drug, 'microbe:', args.microbe, 'disease:', args.disease)
        print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())

        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'Acc']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    # Function to create a formatted print statement
    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)

        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)

            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)

        ret = ret[:-2] + '  '
        return ret

    # Function to perform external testing
    def external_test_run(self):
        self.prepareModel()
        log('Model Prepared')

        if args.load_model != None:
            self.loadModel()

        reses = self.testEpoch()

        log(self.makePrint('Test', args.epoch, reses, True))

        return reses['Acc']

    # Function to train and evaluate the model
    def run(self):
        self.prepareModel()
        log('Model Prepared')
        if args.load_model != None:
            self.loadModel()
            stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            log('Model Initialized')

        best_metrics = {
            'Acc': -float('inf'),
            'Precision': -float('inf'),
            'Recall': -float('inf'),
            'F1': -float('inf'),
            'AUC': -float('inf'),
            'AUPR': -float('inf'),
            'hits@1': -float('inf'),
            'hits@3': -float('inf'),
            'hits@5': -float('inf'),
            'ndcg@1': -float('inf'),
            'ndcg@3': -float('inf'),
            'ndcg@5': -float('inf'),
            'Specificity': -float('inf')
        }
        best_result = None
        best_epoch = -1
        patience = 20
        no_improvement_counter = 0
        self.epoch_metrics = []
        last_best_score = -float('inf')

        for ep in range(stloc, args.epoch):
            tstFlag = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch()
            train_loss = reses
            log(self.makePrint('Train', ep, reses, tstFlag))

            if tstFlag:
                test_r = self.testEpoch()
                log(self.makePrint('Test', ep, reses, tstFlag))

                self.epoch_metrics.append({
                    'epoch': ep,
                    'metrics': test_r.copy()
                })

                current_score = (
                        test_r['hits@1'] * 0.3 +
                        test_r['AUC'] * 0.2 +
                        test_r['AUPR'] * 0.2 +
                        test_r['Acc'] * 0.15 +
                        test_r['F1'] * 0.15
                )

                if current_score > last_best_score:
                    no_improvement_counter = 0
                    last_best_score = current_score
                    best_epoch = ep
                    best_result = test_r.copy()
                    best_result['epoch'] = ep

                    # if test_r['hits@1'] > best_metrics['hits@1']:
                    #     patience_counter = 0
                    #     best_epoch = ep
                    #     best_result = test_r.copy()

                    for metric in best_metrics.keys():
                        best_metrics[metric] = test_r[metric]

                    self.save_model(f'best_model_iteration_{config["iteration"]}')

                    print(f'\nNew best result at epoch {ep}:')
                    print(f"Accuracy: {test_r['Acc']:.6f}")
                    print(f"Precision: {test_r['Precision']:.6f}")
                    print(f"Recall: {test_r['Recall']:.6f}")
                    print(f"F1: {test_r['F1']:.6f}")
                    print(f"AUC: {test_r['AUC']:.6f}")
                    print(f"AUPR: {test_r['AUPR']:.6f}")
                    print(f"hits@1: {test_r['hits@1']:.6f}")
                    print(f"hits@3: {test_r['hits@3']:.6f}")
                    print(f"hits@5: {test_r['hits@5']:.6f}")
                    print(f"ndcg@1: {test_r['ndcg@1']:.6f}")
                    print(f"ndcg@3: {test_r['ndcg@3']:.6f}")
                    print(f"ndcg@5: {test_r['ndcg@5']:.6f}")
                    print(f"Specificity: {test_r['Specificity']:.6f}")
                    print(f"Current Score: {current_score:.6f}")
            else:
                no_improvement_counter += 1
                print(f'\nNo improvement for {no_improvement_counter} evaluations')
                print(f'Current score: {current_score:.6f}, Best score: {last_best_score:.6f}')
                # else:
                #     patience_counter += 1

            if no_improvement_counter >= patience:
                print(f'\nEarly stopping at epoch {ep}')
                print(f'Best epoch was {best_epoch}')
                break

            logs = {
                'loss_all': train_loss['Loss'],
                'loss_pre': train_loss['preLoss'],
                'test_acc': test_r['Acc'],
                'test_precision': test_r['Precision'],
                'test_recall': test_r['Recall'],
                'test_f1': test_r['F1'],
                'test_auc': test_r['AUC'],
                'test_aupr': test_r['AUPR'],
                'test_hits@1': test_r['hits@1'],
                'test_hits@3': test_r['hits@3'],
                'test_hits@5': test_r['hits@5'],
                'test_ndcg@1': test_r['ndcg@1'],
                'test_ndcg@3': test_r['ndcg@3'],
                'test_ndcg@5': test_r['ndcg@5'],
                'test_specificity': test_r['Specificity'],
                'best_acc': best_metrics['Acc'],
                'best_precision': best_metrics['Precision'],
                'best_recall': best_metrics['Recall'],
                'best_f1': best_metrics['F1'],
                'best_auc': best_metrics['AUC'],
                'best_aupr': best_metrics['AUPR'],
                'best_hits@1': best_metrics['hits@1'],
                'best_hits@3': best_metrics['hits@3'],
                'best_hits@5': best_metrics['hits@5'],
                'best_ndcg@1': best_metrics['ndcg@1'],
                'best_ndcg@3': best_metrics['ndcg@3'],
                'best_ndcg@5': best_metrics['ndcg@5'],
                'best_specificity': best_metrics['Specificity'],
                'current_score': current_score,
                'best_score': last_best_score
            }
            wandb.log(logs)

        print('\nTraining finished!')
        print(f'Best epoch: {best_epoch}')
        print('Best results:')
        for metric, value in best_metrics.items():
            print(f'{metric}: {value:.6f}')
        print(f'Best Score: {last_best_score:.6f}')

        best_result['best_epoch'] = best_epoch
        best_result['best_score'] = last_best_score
        return best_result, self.epoch_metrics

    # Function to prepare the model and optimizer
    def prepareModel(self):
        # self.model = Model().cuda()
        self.model = DHGNN()
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

    # Function to train a single epoch
    def trainEpoch(self):
        self.model.train()
        trnLoader = self.handler.trnLoader
        epLoss, epPreLoss = 0, 0
        steps = trnLoader.dataset.__len__() // args.batch

        for i, tem in enumerate(trnLoader):
            drugs, microbes, diseases, labels = tem
            drugs = drugs.long()
            microbes = microbes.long()
            diseases = diseases.long()
            labels = labels.long()

            ceLoss, sslLoss = self.model.calcLosses(
                drugs, microbes, diseases, labels,
                self.handler.torchBiAdj, args.keepRate)

            sslLoss = sslLoss * args.ssl_reg
            regLoss = calcRegLoss(self.model) * args.reg
            loss = ceLoss + regLoss + sslLoss

            epLoss += loss.item()
            epPreLoss += ceLoss.item()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['preLoss'] = epPreLoss / steps
        return ret

    # Function to test a single epoch
    def testEpoch(self):
        self.model.eval()
        tstLoader = self.handler.tstLoader
        total_correct = 0
        total_samples = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for i, tem in enumerate(tstLoader):
                drugs, microbes, diseases, labels = tem
                drugs = drugs.long()
                microbes = microbes.long()
                diseases = diseases.long()
                labels = labels.long()

                pre = self.model.predict(self.handler.torchBiAdj, drugs, microbes, diseases)
                pre = F.log_softmax(pre, dim=1)
                predictions = pre.data.max(1)[1]

                all_preds.append(pre)
                all_labels.append(labels)

                correct = (predictions == labels).sum().item()
                batch_size = labels.size(0)

                total_correct += correct
                total_samples += batch_size

                # batch_acc = correct / batch_size
                # print(f"Batch {i}: {correct}/{batch_size} = {batch_acc:.4f}")

        final_acc = total_correct / total_samples if total_samples > 0 else 0

        assert 0 <= final_acc <= 1, f"Accuracy {final_acc} is out of valid range [0,1]"

        pred_val = torch.cat(all_preds, dim=0)
        val_data = torch.cat(all_labels, dim=0)

        pred_val = pred_val.cpu().numpy()
        val_data = val_data.cpu().numpy()

        hits_1, ndcg_1 = hit_ndcg_value(pred_val, val_data, args.top_1)
        hits_3, ndcg_3 = hit_ndcg_value(pred_val, val_data, args.top_3)
        hits_5, ndcg_5 = hit_ndcg_value(pred_val, val_data, args.top_5)

        metrics = get_metrics(val_data, pred_val)
        aupr, auc, f1, acc, recall, spec, prec = metrics

        print(f'val:{len(val_data)} hits_1:{hits_1:.6f}, ndcg_1:{ndcg_1:.6f}')
        print(f'val:{len(val_data)} hits_3:{hits_3:.6f}, ndcg_3:{ndcg_3:.6f}')
        print(f'val:{len(val_data)} hits_5:{hits_5:.6f}, ndcg_5:{ndcg_5:.6f}')
        print(f'AUPR:{aupr:.6f}, AUC:{auc:.6f}, F1:{f1:.6f}')
        print(f'ACC:{acc:.6f}, Recall:{recall:.6f}, Spec:{spec:.6f}, Prec:{prec:.6f}')
        print('\n')

        ret = {
            'Acc': final_acc,
            'hits@1': hits_1,
            'hits@3': hits_3,
            'hits@5': hits_5,
            'ndcg@1': ndcg_1,
            'ndcg@3': ndcg_3,
            'ndcg@5': ndcg_5,
            'AUPR': aupr,
            'AUC': auc,
            'F1': f1,
            'Recall': recall,
            'Specificity': spec,
            'Precision': prec
        }

        return ret

    # Function to load a pre-trained model
    def loadModel(self):
        self.model.load_state_dict(t.load('../Models/' + args.load_model + '.pkl'))
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        log('Model Loaded')

    # Function to save the trained model
    def save_model(self, model_path):
        model_parent_path = os.path.join(wandb.run.dir, 'ckl')
        if not os.path.exists(model_parent_path):
            os.mkdir(model_parent_path)

        t.save(self.model.state_dict(), '{}/{}_model.pkl'.format(model_parent_path, model_path))


# Main execution block
if __name__ == '__main__':
    args.is_debug = True

    if args.is_debug is True:
        print("DEBUGGING MODE - Start without wandb")
        wandb.init(mode="disabled")
    else:
        wandb.init(project='HC', config=args)
        wandb.run.log_code(".")

    use_cuda = args.gpu >= 0 and t.cuda.is_available()
    device = 'cuda:{}'.format(args.gpu) if use_cuda else 'cpu'
    if use_cuda:
        t.cuda.set_device(device)
    args.device = device

    logger.saveDefault = True

    log('Start processing data')
    handler = DataHandler()
    handler.LoadData()
    log('Load Data finished')

    coach = Coach(handler)
    config = dict()

    all_metrics = [
        'TestAcc', 'TestPrecision', 'TestRecall', 'TestF1', 'TestAUC', 'TestAUPR',
        'hits@1', 'hits@3', 'hits@5',
        'ndcg@1', 'ndcg@3', 'ndcg@5',
        'Specificity'
    ]
    all_results = {metric: [] for metric in all_metrics}

    all_epoch_results = []
    iteration_best_results = []

    for i in range(args.iteration):
        print('{}-th iteration'.format(i + 1))

        seed = args.seed + i
        config['seed'] = seed
        config['iteration'] = i + 1
        set_seed(seed)

        if args.data == 'LINCS':
            iteration_result, epoch_metrics = coach.external_test_run()
        else:
            iteration_result, epoch_metrics = coach.run()

        result_mapped = {
            'TestAcc': iteration_result['Acc'],
            'TestPrecision': iteration_result['Precision'],
            'TestRecall': iteration_result['Recall'],
            'TestF1': iteration_result['F1'],
            'TestAUC': iteration_result['AUC'],
            'TestAUPR': iteration_result['AUPR'],
            'hits@1': iteration_result['hits@1'],
            'hits@3': iteration_result['hits@3'],
            'hits@5': iteration_result['hits@5'],
            'ndcg@1': iteration_result['ndcg@1'],
            'ndcg@3': iteration_result['ndcg@3'],
            'ndcg@5': iteration_result['ndcg@5'],
            'Specificity': iteration_result['Specificity']
        }

        current_result = [result_mapped[metric] for metric in all_metrics]
        iteration_best_results.append(current_result)

        for metric in all_metrics:
            all_results[metric].append(result_mapped[metric])

        print(f'\nIteration {i + 1} Results:')
        print('-' * 30)
        for metric in all_metrics:
            print(f'{metric:<15} = {result_mapped[metric]:.6f}')

        for epoch_data in epoch_metrics:
            all_epoch_results.append({
                'iteration': i + 1,
                'epoch': epoch_data['epoch'],
                'metrics': epoch_data['metrics']
            })

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    experiment_name = f"{args.data}_{timestamp}"
    results_parent_path = os.path.join('Results', experiment_name)
    if not os.path.exists(results_parent_path):
        os.makedirs(results_parent_path)

    with open(os.path.join(results_parent_path, 'metrics_all.txt'), 'w', encoding='utf-8') as f:
        f.write(f'Iteration: {args.iteration}\n\n')

        header = 'Iteration\tEpoch\t' + '\t'.join(all_metrics)
        f.write(header + '\n')

        for result in all_epoch_results:
            metrics_values = [f"{result['metrics'][metric.replace('Test', '')]:.6f}" for metric in all_metrics]
            row = f"{result['iteration']}\t{result['epoch']}\t" + '\t'.join(metrics_values)
            f.write(row + '\n')

        f.write('\nBest Results for Each Iteration:\n')
        f.write('Iteration\t' + '\t'.join(all_metrics) + '\n')
        for i, best_result in enumerate(iteration_best_results):
            metrics_values = [f"{value:.6f}" for value in best_result]
            row = f"{i + 1}\t" + '\t'.join(metrics_values)
            f.write(row + '\n')

    results_array = np.array(iteration_best_results)
    avg_r = np.mean(results_array, axis=0)

    print('\nFinal Results:')
    print('=' * 50)
    for i, metric in enumerate(all_metrics):
        print(f'{metric:<15} = {avg_r[i]:.6f}')

    iteration_best_results.append(avg_r)

    with open(os.path.join(results_parent_path, 'metrics_info.txt'), 'w') as f:
        for i, metric in enumerate(all_metrics):
            f.write(f'Column {i}: {metric}\n')

    with open(os.path.join(results_parent_path, 'config.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')

    print(f'\nResults saved to: {results_parent_path}')
    print('result saved!!!')

    wandb.finish()
