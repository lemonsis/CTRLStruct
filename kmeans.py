import torch
from Model import Model
import torch.nn as nn
from transformers import BartTokenizer, BertTokenizer
import numpy as np
from kmeans_pytorch import kmeans, kmeans_predict
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE 
from sklearn import metrics

device='cuda:1'
CHECKPOINT_PATH = './model/0.2/pc_best_model_40.pth'
TRAINING_PATH = 'data/training_data1.txt'
TESTING_PATH = 'test/test_personachat1.txt'
VALID_PATH = 'valid/valid_personachat1.txt'

SAVE_TRAINING_EMB_PATH = "kmeans/0.2_cluster_0_20/sentence_embedding_0_20.pth"
SAVE_TEST_EMB_PATH = "kmeans/0.2_pc_cluster_40/test_sentence_embedding_40.pth"
SAVE_VALID_EMB_PATH = "kmeans/0.2_pc_cluster_40/valid_sentence_embedding_40.pth"

CLUSTER_CENTER_PATH = "kmeans/0.2_pc_cluster_40/cluster_centers_60.pth"
CLUSTER_ID_PATH = "kmeans/0.2_cluster_0_20/cluster_ids_x_60.pth"

TSNE_PATH = "kmeans/0.2_pc_cluster_40/pc_tsne_60.npy"
SAVEFIG_PATH = "kmeans/0.2_pc_cluster_40/pc_kmeans_tsne_60.png"

TEST_CLUSTER_ID_PATH = 'kmeans/0.2_pc_cluster_40/test_cluster_ids_60.pth'
VALID_CLUSTER_ID_PATH = 'kmeans/0.2_pc_cluster_40/valid_cluster_ids_60.pth'

def load_model():
    model = Model().to(device)
    model = nn.DataParallel(model, device_ids=[1, 2])
    model = model.to(device)
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    # with torch.no_grad():
    #     with open(TRAINING_PATH, 'r') as f:
    #         sentence = f.readlines()
    #         dim = 1024
    #         # size = len(sentence)
    #         data = torch.empty([0, dim], dtype=torch.float32, device=device)
    #         i=0
    #         for sen in tqdm(sentence):
    #             i += 1
    #             encode_input = tokenizer(sen, return_tensors='pt')
    #             encode_output = model(**encode_input)
    #             data = torch.cat((data, encode_output), 0)
    #             if i % 100 == 0:
    #                 print(data.shape)
    # torch.save(data, SAVE_TRAINING_EMB_PATH)

    with torch.no_grad():
        with open(TESTING_PATH, 'r') as f:
            sentence = f.readlines()
            dim = 1024
            # size = len(sentence)
            data = torch.empty([0, dim], dtype=torch.float32, device=device)
            i=0
            for sen in tqdm(sentence):
                i += 1
                encode_input = tokenizer(sen, return_tensors='pt')
                encode_output = model(**encode_input)
                data = torch.cat((data, encode_output), 0)
                if i % 100 == 0:
                    print(data.shape)
    torch.save(data, SAVE_TEST_EMB_PATH)

    with torch.no_grad():
        with open(VALID_PATH, 'r') as f:
            sentence = f.readlines()
            dim = 1024
            # size = len(sentence)
            data = torch.empty([0, dim], dtype=torch.float32, device=device)
            i=0
            for sen in tqdm(sentence):
                i += 1
                encode_input = tokenizer(sen, return_tensors='pt')
                encode_output = model(**encode_input)
                data = torch.cat((data, encode_output), 0)
                if i % 100 == 0:
                    print(data.shape)
    torch.save(data, SAVE_VALID_EMB_PATH)

def kmeans_cluster(data, num_clusters):
    cluster_ids_x, cluster_centers = kmeans(X=data, num_clusters=num_clusters, distance='cosine', device=device)
    torch.save(cluster_centers, CLUSTER_CENTER_PATH)
    torch.save(cluster_ids_x, CLUSTER_ID_PATH)

def kmeans_cluster_predict(data, cluster_centers):
    cluster_centers = torch.load(CLUSTER_CENTER_PATH)
    cluster_ids_y = kmeans_predict(data, cluster_centers, 'cosine', device=device)
    torch.save(cluster_ids_y, VALID_CLUSTER_ID_PATH)
    torch.set_printoptions(profile="full")
    print(cluster_ids_y)
    return cluster_ids_y

def tsne(data, cluster_centers, num_clusters):
    data = data.cpu().numpy()
    cluster_centers = cluster_centers.cpu().numpy()
    data = np.vstack((data, cluster_centers))
    sen_tsne = TSNE(n_components=2, init='random', n_jobs=-1).fit_transform(data)
    np.save(TSNE_PATH, sen_tsne)
    center = sen_tsne[-num_clusters:, :]
    x = sen_tsne[:-num_clusters, :]
    return x, center

def plot(x, cluster_ids_x, center):
    cluster_ids_x = cluster_ids_x.cpu().numpy()
    plt.figure(figsize=(8, 6), dpi=260)
    plt.scatter(x[:, 0], x[:, 1], s=1, c=cluster_ids_x, cmap='cool')
    # plt.scatter(y[:, 0], y[:, 1], c=cluster_ids_y, cmap='cool', marker='X')
    plt.scatter(
        center[:, 0], center[:, 1],
        s=5,
        c='white',
        alpha=0.6,
        edgecolors='black',
        linewidths=2
    )
    plt.tight_layout()
    plt.savefig(SAVEFIG_PATH)

def evaluation(data, cluster_ids_x):
    data = data.cpu().numpy()
    cluster_ids_x = cluster_ids_x.cpu().numpy()
    print("calinski_harabasz_score:")
    print(metrics.calinski_harabasz_score(data, cluster_ids_x))
    print("DBI:")
    print(metrics.davies_bouldin_score(data, cluster_ids_x))
    print("Silhouette Coefficient:")
    print(metrics.silhouette_score(data, cluster_ids_x, metric='cosine'))



# load_model()
# flag = "valid"
# if flag == "train":
#     data = torch.load(SAVE_TRAINING_EMB_PATH)
#     kmeans_cluster(data, 60)
#     cluster_center = torch.load(CLUSTER_CENTER_PATH)
#     cluster_ids_x = torch.load(CLUSTER_ID_PATH)
#     x, center = tsne(data, cluster_center, 60)
#     plot(x, cluster_ids_x, center)
#     evaluation(data, cluster_ids_x)
# elif flag == "valid":
#     data = torch.load(SAVE_VALID_EMB_PATH)
#     cluster_centers = torch.load(CLUSTER_CENTER_PATH)
#     kmeans_cluster_predict(data, cluster_centers)



# data = torch.load("sentence_embedding_0_20.pth")
# # cluster_ids_x = torch.load("cluster_0_20/cluster_ids_x_50.pth")
# # cluster_centers = torch.load("cluster_0_20/cluster_centers_50.pth")
# data = data.cpu().numpy()
# sen_tsne = TSNE(n_components=2, init='random', n_jobs=-1).fit_transform(data)
# np.save("tsne.npy", sen_tsne)
# sen_tsne = torch.from_numpy(sen_tsne)
# cluster_ids_x, cluster_centers = kmeans(
#     X=sen_tsne, num_clusters=50, distance='cosine', device=device
# )
# torch.save(cluster_centers, "cluster_centers_tsne.pth")
# torch.save(cluster_ids_x, "cluster_ids_x_tsne.pth")

# plt.figure(figsize=(8, 6), dpi=260)
# plt.scatter(sen_tsne[:, 0], sen_tsne[:, 1], s=1, c=cluster_ids_x, cmap='cool')
# # plt.scatter(y[:, 0], y[:, 1], c=cluster_ids_y, cmap='cool', marker='X')
# plt.scatter(
#     cluster_centers[:, 0], cluster_centers[:, 1],
#     s=5,
#     c='white',
#     alpha=0.6,
#     edgecolors='black',
#     linewidths=2
# )
# plt.tight_layout()
# plt.savefig("kmeans_tsne.png")
