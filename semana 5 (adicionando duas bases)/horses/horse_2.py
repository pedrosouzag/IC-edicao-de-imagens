import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F 

import kagglehub

# Download latest version
path = kagglehub.dataset_download("sanikamal/horses-or-humans-dataset")

#print("Path to dataset files:", path)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

treino = datasets.ImageFolder(
    root=f'{path}/horse-or-human/train',
    transform=transform
    #transform= transforms.ToTensor()
)

teste = datasets.ImageFolder(
    root=f'{path}/horse-or-human/validation',
    transform=transform
    #transform= transforms.ToTensor()
)

#print(len(treino)) # 1027
#print(len(teste)) # 256

train_loader = DataLoader(treino, batch_size = 32, shuffle = True)
test_loader = DataLoader(teste, batch_size = 32, shuffle = False)


# imagens, labels = next(iter(train_loader))
# print(imagens.shape)

# torch.Size([32, 3, 300, 300])


# pega uma parte da imagem, roda o kernel, multiplica a joga para uma matriz, depois nessa matriz pega os pares e mantem o maior numero.
#Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        #camadas convolucional 1
        # 3 canais
        self.conv1 = nn.Conv2d(3, 32,kernel_size=5) # 300 - 5 + 1 = 296 / 2 = 148 >> 32 matrizes 148 x 148
        #camada convolucional 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5) # 148 - 5 + 1 = 144 / 2 = 72 >> 64 matrizes 72 x 72 

        #dropout
        self.dropout = nn.Dropout2d()


        # camada fully connected 1 (linear)
        self.fc1 = nn.Linear(64 * 72 * 72 , 256)

        # camada fully connected 2 (linear)
        self.fc2 = nn.Linear(256, 128)

        # camada fully connected 3 (linear)
        self.fc3 = nn.Linear(128, 2)


    def forward(self, x):
        # camada conv1: [32, 3, 300, 300] para [32, 32 ,148,148]
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)

        # camada conv2 + dropout: [64,16,13,13] para [32, 64 ,72 ,72 ]
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.dropout(x)

        # flatten: [32, 64 ,72 ,72 ] para [32,64 * 72 * 72]
        x = x.view(-1, 64 * 72 * 72 )

        # fc1:  64 * 72 * 72  para 256 
        x = F.relu(self.fc1(x))

        # fc2: 256 para 128 
        x = F.relu(self.fc2(x))

        # fc3: 128 para 2 
        x = self.fc3(x)

        return x


# usa GPU se disponível, senão usa CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNN().to(device)

# optmizner escolhido para evitar ruido, zigue zag etc
optimizer = optim.Adam(model.parameters(), lr=0.001)

#funcao de erro
loss_fn = nn.CrossEntropyLoss()


def train(epoch) :
    model.train()
    for idBatch, (images, labels) in enumerate(train_loader):

        # manda as imagens e respostas para GPU ou CPU
        images, labels = images.to(device), labels.to(device)

        # zera o gradiente, db e dw
        optimizer.zero_grad()

        # chama o foward e recbe o [64,10]
        # output = model.forward(images)
        output = model(images)

        # chama a funcao para calcular o erro
        loss = loss_fn(output, labels)

        # com o erro volta atualizando os pesos
        loss.backward()

        # otimiza os pesos, para evitar zig zag, ruidos etc
        optimizer.step()

        if idBatch % 100 == 0:
            # pega a predicao (numero com maior confianca)
            pred = output.argmax(dim=1)
            
            print(f'Epoch {epoch} [{idBatch * len(images)}/{len(train_loader.dataset)}] '
                f'Loss: {loss.item():.4f} | '
                f'Esperado: {labels[0].item()} | '
                f'Previsto: {pred[0].item()}')

#def test (epoch):
def test():

    model.eval() # dessliga o dropout
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for idBatch, (images, labels) in enumerate(test_loader):

            # manda as imagens e respostas para GPU ou CPU
            images, labels = images.to(device), labels.to(device)

            ## chama o foward e recebe o [64,10]
            output = model(images)

            # chama a funcao para calcular o erro e soma para fazer a media
            test_loss += loss_fn(output, labels).item()

            # pega a predicao (numero com maior confianca) e verifica pra somar os acertos
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()


    # loss medio por imagem
    test_loss /= len(test_loader.dataset)

    #print (f'Epoch {epoch} ')
    print('Teste Resultados:')
    print(f'Loss medio: {test_loss:.4f} | ' f'Acuracia: {correct}/{len(test_loader.dataset)} '
        f'({100. * correct / len(test_loader.dataset):.1f}%)')



# principal
for epoch in range(1,11 ):
    train(epoch)
    #test(epoch)

test()

          

