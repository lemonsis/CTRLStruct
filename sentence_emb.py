from Model import Model
import argparse
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from transformers import BartTokenizer
import random
# import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
from datasets import load_dataset
# import nlpaug.augmenter.sentence as nas
# import nlpaug.flow as nafc
# from nlpaug.util import Action
# from transformers import BertTokenizer


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--max_length", type=int, default=64, help="sentence max length")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--epochs", type=int, default=20, help="epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--tao", type=float, default=0.05, help="temperature")
    parser.add_argument("--device", type=str, default='cuda:0', help="device")
    parser.add_argument("--display_interval", type=int, default=10, help="display interval")
    parser.add_argument("--save_interval", type=int, default=10, help="save interval")
    parser.add_argument("--dropout_rate", type=float, default=0.25, help="dropout_rate")
    parser.add_argument("--Lambda", type=float, default=0.2, help="weak relativity")
    parser.add_argument("--dialogue_file", type=str, default='./data/personachat.txt', help="dialogue_file path")
    parser.add_argument("--pretrained_model_path", type=str, default='facebook/bart-large', help="pretrained_model_path")
    parser.add_argument("--best_model_path", type=str, default='./model/0.2/pc_best_model_20.pth', help="best_model_path")
    args = parser.parse_args()
    return args

def read_data(args):
    with open(args.dialogue_file, 'r') as f:
        sentences = f.readlines()
    dl = DataLoader(sentences,
                    batch_size=args.batch_size,
                    collate_fn=duplicate_batch,
                    num_workers=16,
                    pin_memory=True,
                    drop_last=True)
    dl_label = DataLoader(sentences,
                        batch_size=args.batch_size,
                        num_workers=16,
                        drop_last=True)
    return dl, dl_label

def dropout_augmentation(batch, idx):
    item = batch[idx-1]
    batch.insert(idx, item)
    return batch

def insert_augmentation(batch, idx):
    item = batch[idx-1]
    if item == '<eod>\n':
        batch.insert(idx, '<eod>\n')
    else:
        aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', 
                                        action="insert")
        augmented_text = aug.augment(item)
        batch.insert(idx, augmented_text)
    return batch

def replace_augmentation(batch, idx):
    item = batch[idx-1]
    if item == '<eod>\n':
        batch.insert(idx, '<eod>\n')
    else:
        aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', 
                                        action="substitute")
        augmented_text = aug.augment(item)
        batch.insert(idx, augmented_text)
    return batch

def synonym_augmentation(batch, idx):
    item = batch[idx-1]
    if item == '<eod>\n':
        batch.insert(idx, '<eod>\n')
    else:
        aug = naw.SynonymAug(aug_src='wordnet')
        augmented_text = aug.augment(item)
        batch.insert(idx, augmented_text)
    return batch

def duplicate_batch(batch):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    random.seed(123)
    length = len(batch)
    for sentence in range(1, 2 * length, 2):
        num = random.randint(1, 4)
        if num == 1:
            batch = dropout_augmentation(batch, sentence)
        elif num == 2:
            batch = insert_augmentation(batch, sentence)
        elif num == 3:
            batch = replace_augmentation(batch, sentence)
        else:
            batch = synonym_augmentation(batch, sentence)
    batch_encoding = tokenizer(batch,
                               padding=True,
                               truncation=True,
                               max_length=64,
                               return_tensors='pt')
    return batch_encoding

def abs_correlation_loss(y_pred, args):
    idxs = torch.arange(0, y_pred.shape[0], device=args.device)
    y_true = idxs + 1 - idxs % 2 * 2
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    similarities = similarities - torch.eye(y_pred.shape[0], device=args.device) * 1e12
    similarities = similarities / args.tao
    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)

def strong_relativity_loss(y_pred, batch, args):
    y_true = []
    for i in range(len(batch) - 2):
        if batch[i] == '<eod>\n' and batch[i+1] == '<eod>\n':
            y_true.append(i+1)
        elif batch[i] == '<eod>\n' and batch[i+1] != '<eod>\n':
            y_true.append(i-1)
        elif batch[i+2] != '<eod>\n':
            y_true.append(i+2)
        elif batch[i+2] == '<eod>\n' and batch[i+1] != '<eod>\n':
            y_true.append(i+1)
        elif batch[i+2] == '<eod>\n' and batch[i+1] == '<eod>\n':
            y_true.append(i-1)
    y_true.append(len(batch) - 1)
    y_true.append(len(batch) - 2)
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    similarities = similarities - torch.eye(y_pred.shape[0], device=args.device) * 1e12
    similarities = similarities / args.tao
    loss = F.cross_entropy(similarities, torch.tensor(y_true, dtype=int).to(args.device))
    return torch.mean(loss)
    
def weak_relativity_loss(y_pred, batch, args):
    y_true = [1, 0]
    for i in range(2, len(batch) - 1):
        if batch[i] == '<eod>\n' and batch[i+1] == '<eod>\n':
            y_true.append(i+1)
        elif batch[i] == '<eod>\n' and batch[i+1] != '<eod>\n':
            y_true.append(i-1)
        elif batch[i-2] != '<eod>\n':
            y_true.append(i-2)
        elif batch[i-2] == '<eod>\n' and batch[i-1] == '<eod>\n':
            y_true.append(i+1)
        elif batch[i-2] == '<eod>\n' and batch[i-1] != '<eod>\n':
            y_true.append(i-1)
    if batch[-1] == '<eod>\n':
        y_true.append(len(batch) - 2)
    elif batch[-1] != '<eod>\n' and batch[-3] == '<eod>\n':
        y_true.append(len(batch) - 2)
    elif batch[-1] != '<eod>\n' and batch [-3] != '<eod>\n':
        y_true.append(len(batch) - 3)
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    similarities = similarities - torch.eye(y_pred.shape[0], device=args.device) * 1e12
    similarities = similarities / args.tao
    loss = F.cross_entropy(similarities, torch.tensor(y_true, dtype=int).to(args.device))
    return torch.mean(args.Lambda * loss)

def train(args):
    dl, dl_label = read_data(args)
    model = Model(args.pretrained_model_path).to(args.device)
    model = nn.DataParallel(model, device_ids=[0,1,2, 3, 4, 5,6,7])
    model.to(args.device)
    # checkpoint = torch.load('model/simcse_style/pc_best_model_20.pth')
    # model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    batch_idx = 0
    min_loss = 10000
    for epoch_idx in range(args.epochs):
        epoch_losses = []
        for data, data_label in tqdm(zip(dl, dl_label)):
            batch_idx += 1
            new_data_label = []
            for i in range(args.batch_size):
                new_data_label.append(data_label[i])
                new_data_label.append(data_label[i])
            pred = model(input_ids=data["input_ids"].to(args.device),
                        attention_mask=data["attention_mask"].to(args.device))
            loss1 = abs_correlation_loss(pred, args)
            loss2 = strong_relativity_loss(pred, new_data_label, args)
            loss3 = weak_relativity_loss(pred, new_data_label, args)
            loss = loss1 + loss2 + loss3
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()
            epoch_losses.append(loss)
            if batch_idx % args.display_interval == 0:
                logging.getLogger().setLevel(logging.INFO)
                logging.info(f"epoch: {epoch_idx}, batch_idx: {batch_idx}, loss: {loss:>10f}")
        avg_epoch_loss = np.mean(epoch_losses)
        if avg_epoch_loss < min_loss:
            print(avg_epoch_loss,'<', min_loss," new model saved")
            min_loss = avg_epoch_loss
            torch.save({
                'epoch': epoch_idx,
                'model_state_dict': model.state_dict(),
                'loss': avg_epoch_loss
            }, args.best_model_path)

def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
