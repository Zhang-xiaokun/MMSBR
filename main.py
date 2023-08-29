import argparse
import pickle
import time
from util import Data, split_validation, init_seed
from model import *
import os


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Cell_Phones_and_Accessories', help='dataset name: Office_Products/Grocery_and_Gourmet_Food/Cell_Phones_and_Accessories/Clothing_Shoes_and_Jewelry/Home_and_Kitchen/sample')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size 512')
parser.add_argument('--embSize', type=int, default=64, help='embedding size')
parser.add_argument('--imgEmbSize', type=int, default=64, help='image embedding size 64')
parser.add_argument('--textEmbSize', type=int, default=64, help='text embedding size 64')
parser.add_argument('--featureEmbSize', type=int, default=64, help='feature embedding size 64')
parser.add_argument('--layer', type=float, default=6, help='the number of stacked layers for hierarchical pivot transformer (double for text and image)')
parser.add_argument('--num_heads', type=int, default=4, help='number of attention heads')
parser.add_argument('--feature_num', type=int, default=4, help='the number of generated features')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--num_negatives', type=int, default=100, help='the number of negatives 100')
parser.add_argument('--lam', type=float, default=0.01, help='ssl task maginitude 0.01')
parser.add_argument('--filter', type=bool, default=False, help='filter incidence matrix')

opt = parser.parse_args()
print(opt)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
torch.cuda.set_device(0)

def main():
    # data formulation list: 0:id_seq, 1:price_seq, 2:rating_seq, 3:cate_seq, 4:price_list, 5:rating_list, 6:cate_list, 7:labs
    train_data = pickle.load(open('./datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('./datasets/' + opt.dataset + '/test.txt', 'rb'))
    # init_seed(2022, True)

    if opt.dataset == 'Grocery_and_Gourmet_Food':
        n_node = 11638
        n_price = 100
        n_category = 665
    elif opt.dataset == 'Cell_Phones_and_Accessories':
        n_node = 8614
        n_price = 69
        n_category = 48
    elif opt.dataset == 'Sports_and_Outdoors':
        n_node = 18796
        n_price = 99
        n_category = 1259
    else:
        print("unkonwn dataset")
    # data_formate: sessions, price_seq, matrix_session_item matrix_pv
    train_data = Data(train_data, shuffle=True, n_node=n_node, n_price=n_price)
    test_data = Data(test_data, shuffle=True, n_node=n_node, n_price=n_price)
    model = trans_to_cuda(Beyond(price_list=train_data.price_list, category_list=train_data.cate_list, n_node=n_node, n_price=n_price, n_category=n_category, lr=opt.lr, layers=opt.layer,  feature_num=opt.feature_num, l2=opt.l2, lam=opt.lam, dataset=opt.dataset, num_heads=opt.num_heads, emb_size=opt.embSize, img_emb_size=opt.imgEmbSize, text_emb_size=opt.textEmbSize, feature_emb_size=opt.featureEmbSize, batch_size=opt.batchSize, num_negatives=opt.num_negatives))

    top_K = [1, 5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0, 0]
        best_results['metric%d' % K] = [0, 0, 0]

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        metrics, total_loss = train_test(model, train_data, test_data)
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            metrics['ndcg%d' % K] = np.mean(metrics['ndcg%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
            if best_results['metric%d' % K][2] < metrics['ndcg%d' % K]:
                best_results['metric%d' % K][2] = metrics['ndcg%d' % K]
                best_results['epoch%d' % K][2] = epoch
        print(metrics)
        # for K in top_K:
        #     print('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f\tNDCG%d: %.4f\tEpoch: %d,  %d, %d' %
        #           (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],K, best_results['metric%d' % K][2],
        #            best_results['epoch%d' % K][0], best_results['epoch%d' % K][1], best_results['epoch%d' % K][2]))
        print('P@1\tP@5\tM@5\tN@5\tP@10\tM@10\tN@10\tP@20\tM@20\tN@20\t')
        print("%.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f" % (
            best_results['metric1'][0], best_results['metric5'][0], best_results['metric5'][1],
            best_results['metric5'][2], best_results['metric10'][0], best_results['metric10'][1],
            best_results['metric10'][2], best_results['metric20'][0], best_results['metric20'][1],
            best_results['metric20'][2]))
        print("%d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d" % (
            best_results['epoch1'][0], best_results['epoch5'][0], best_results['epoch5'][1],
            best_results['epoch5'][2], best_results['epoch10'][0], best_results['epoch10'][1],
            best_results['epoch10'][2], best_results['epoch20'][0], best_results['epoch20'][1],
            best_results['epoch20'][2]))

if __name__ == '__main__':
    main()
