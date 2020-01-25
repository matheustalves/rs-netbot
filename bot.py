# ==============================
# BOT RUNESCAPE COM CONV NET
# Autor: Matheus Teixeira
# ==============================

import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mss import mss
from PIL import Image
from pynput.mouse import Button, Controller
import time

# Usando CUDA para Processamento
device = torch.device("cuda:0")

# Refazer dataset
REBUILD_DATA = False

# Criar Dateset (Classificar imagens)
class Spiders_DATA():
    IMG_SIZE = 50
    SPIDERS = "Dataset/Spiders"
    NOT_SPIDERS = "Dataset/Not_Spiders"
    LABELS = {NOT_SPIDERS: 0, SPIDERS: 1}
    training_data = []
    spidercount = 0
    not_spiderscount = 0

    def make_training_data(self):
        for label in self.LABELS:
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

                    if label == self.SPIDERS:
                        self.spidercount += 1
                    elif label == self.NOT_SPIDERS:
                        self.not_spiderscount += 1

                except Exception as e:
                    pass
        
        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print("Spiders:", self.spidercount)
        print("Not Spiders:", self.not_spiderscount)

# Rede -- 3 Conv Nodes --> 2 Linear Nodes
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 5)

        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)
        self.fc1 = nn.Linear(self._to_linear, 32)
        self.fc2 = nn.Linear(32, 2)
    
    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

# Cria Dataset
if REBUILD_DATA:
    spiderdata = Spiders_DATA()
    spiderdata.make_training_data()

# Load Dataset
training_data = np.load("training_data.npy", allow_pickle=True)

# Processamento e normalização das imgs do Dataset
X = torch.Tensor([i[0] for i in training_data]).view(-1, 50,50)
X = X/255.0

# y = classificação das imgs
y = torch.Tensor([i[1] for i in training_data])

# Inicialização de variáveis
train_X = X
train_y = y
test_X = X
test_y = y
BATCH_SIZE = 10
EPOCHS = 20

# Inicialização da conv net com processamento na GPU
net = Net().to(device)

# Treinamento da conv net
def train(net):
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
            batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,50,50)
            batch_y = train_y[i:i+BATCH_SIZE]

            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            net.zero_grad()

            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()

# Treina a rede
train(net)

# Transforma classificações corretas em Tensors
correto = np.eye(2)[1]
correto = torch.Tensor(correto)
correto = torch.argmax(correto)

mouse = Controller()

# Define canvas do jogo
bbox = {'top': 50, 'left': 30, 'width': 500, 'height': 300}
sct = mss()

template2 = cv2.imread('name.jpg', 0)
w2, h2 = template2.shape[::-1]

cX = 0
cY = 0

# Tempo de execução
t_end = time.time() + 60*1.5

# Execução do bot
while time.time() < t_end:

    # Processamento de imagem do canvas
    sct_img = sct.grab(bbox)
    output = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGR2GRAY)


    # Divide canvas em quadrados 50x50 e os coloca em array
    start_row, start_col = int(0), int(0)
    img_slices = []
    for row in range(0,6):
        end_row = int(50*(1+row))
        for col in range(0,10):
            end_col = int(50*(1+col))
            cropped = output[start_row:end_row, start_col:end_col]
            img_slices.append(cropped)
            start_col = end_col
        start_row = end_row
        start_col = int(0)

    # Transforma quadrados do canvas em Tensors e normaliza
    img_slices = torch.Tensor([i for i in img_slices]).view(-1, 50,50)
    img_slices = img_slices/255.0

    # Execução da rede 
    identificou = False
    with torch.no_grad():
        
        # Verifica se possui objeto no atual quadrado do canvas
        for i in tqdm(range(len(img_slices))):
            net_out = net(img_slices[i].view(-1,1,50,50).to(device))[0]
            predicted_class = torch.argmax(net_out)

            # Se possuir, desenha quadrado no objeto
            if predicted_class == correto:
                identificou = True
                if i < 10:
                    cv2.rectangle(output, (50*i,0), (50*(i+1), 50), (0, 255, 0), 2)
                    cX = 25*(i+1)
                    cY = 25
                elif i < 20:
                    cv2.rectangle(output, (50*(i-10),50), (50*(i-9), 100), (0, 255, 0), 2)
                    cX = 25*(i-9)
                    cY = 50
                elif i < 30:
                    cv2.rectangle(output, (50*(i-20),100), (50*(i-19), 150), (0, 255, 0), 2)
                    cX = 25*(i-19)
                    cY = 75
                elif i < 40:
                    cv2.rectangle(output, (50*(i-30),150), (50*(i-29), 200), (0, 255, 0), 2)
                    cX = 25*(i-29)
                    cY = 100
                elif i < 50:
                    cv2.rectangle(output, (50*(i-40),200), (50*(i-39), 250), (0, 255, 0), 2)
                    cX = 25*(i-39)
                    cY = 125
                else:
                    cv2.rectangle(output, (50*(i-50),250), (50*(i-49), 300), (0, 255, 0), 2)
                    cX = 25*(i-49)
                    cY = 150


    res2 = cv2.matchTemplate(output, template2, cv2.TM_CCOEFF_NORMED)
    threshold2 = 0.69
    loc2 = np.where(res2 >= threshold2)

    em_combate = False
    for pt in zip(*loc2[::-1]):
        if pt is not None:
            em_combate = True 
        cv2.rectangle(output, pt, (pt[0]+w2, pt[1]+h2), (0, 255, 255), 2)            
    
    if em_combate:
        print("Em combate.")
    else:
        print("Fora de combate.")
        if identificou:
        	mouse.position = (cX, cY)
        	mouse.click(Button.left, 1)


    cv2.imshow('screen', output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
