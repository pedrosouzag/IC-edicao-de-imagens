import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F 

from torchvision.models import vgg16, VGG16_Weights

import kagglehub

# Download latest version
path = kagglehub.dataset_download("sanikamal/horses-or-humans-dataset")

#print("Path to dataset files:", path)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

# usa GPU se disponível, senão usa CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = vgg16(weights=VGG16_Weights.DEFAULT) #pega os pesos treinados
for peso in model.parameters():
    peso.requires_grad = False

model.classifier[6] = nn.Linear(model.classifier[6].in_features,2)
model = model.to(device)


# optmizner escolhido para evitar ruido, zigue zag etc
optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.001)

#funcao de erro
loss_fn = nn.CrossEntropyLoss()


def train(epoch) :
    model.train()
    for idBatch, (images, labels) in enumerate(train_loader):

        # manda as imagens e respostas para GPU ou CPU
        images, labels = images.to(device), labels.to(device)

        # zera o gradiente, db e dw
        optimizer.zero_grad()

        # chamada a camada
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

            ## chama o foward e recebe a saida
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
for epoch in range(1,3 ):
    train(epoch)
    #test(epoch)

test()