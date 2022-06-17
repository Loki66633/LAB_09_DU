import argparse
import os
import re
import pkbar
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from models import Model
from Dataset import Dataset

def sample(preds, temperature=1):
    preds = preds / temperature
    exp_preds = torch.exp(preds)
    preds = exp_preds / torch.sum(exp_preds)
    probas = torch.multinomial(preds, 1)
    return probas





cuda = True if torch.cuda.is_available() else False
print(cuda)
if cuda:
    device = torch.device('cuda')


def train(dataset, model, args):
    global device
    cuda = True if torch.cuda.is_available() else False
    criterion = nn.CrossEntropyLoss()

    model_ver = 'Model_LSTM'
    if cuda:
        device = torch.device('cuda')
        model.cuda()
        criterion = criterion.cuda()


    #model.train()
    optimizer = torch.optim.Adadelta(model.parameters())
    dataloader = DataLoader(dataset, batch_size=args.batch_size,num_workers=16, shuffle=True, drop_last=True, pin_memory=True)
    train_per_epoch = int(dataset.__len__() / args.batch_size)


    for epoch in range(args.max_epochs):
        print('\nEpoch: %d/%d' % (epoch + 1, args.max_epochs))
        kbar = pkbar.Kbar(target=train_per_epoch, width=20)
        state_h, state_c = model.init_state(args.sequence_length)

        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            x = x.cuda()
            y = y.cuda()
            state_h =state_h.cuda()
            state_c = state_c.cuda()
            #print("X=",x.is_cuda)
            #print("Y=",y.is_cuda)

            #print("state_h=",state_h.is_cuda)
            #print("state_c=",state_c.is_cuda)
            #print("model=",next(model.parameters()).is_cuda)

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))

            #inp = x.type(torch.LongTensor).to(device)
            #out = y.type(torch.LongTensor).to(device)
            #output = model(inp)

            loss = criterion(y_pred.transpose(1, 2), y.to(device))

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()

            optimizer.step()

            kbar.update(batch, values=[("loss", loss)])
            #print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })
        if(epoch%10 == 0):
            path_str = os.path.join('models/', model_ver + '_epoch_text_' + str(epoch + 1) + '.pth')
            torch.save(model.state_dict(), path_str)

def predict(dataset, model, text, next_words=100):
    model.eval()

    words = text.split(' ')
    state_h, state_c = model.init_state(len(words))
    state_h = state_h.cuda()
    state_c = state_c.cuda()

    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
        x = x.cuda()
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().cpu().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])

    return words


parser = argparse.ArgumentParser()
parser.add_argument('--max-epochs', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--sequence-length', type=int, default=4)
args = parser.parse_args()

dataset = Dataset(args)
model = Model(dataset)
model.cuda()
model_to_load = "models/Model_LSTM_epoch_text_51.pth"
loaded = torch.load(model_to_load)
model.load_state_dict(loaded)
train(dataset, model, args)
nesto = (predict(dataset, model, text=''))
print("")
str= ' '.join(nesto)
print(str[1:].capitalize())


