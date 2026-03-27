from simple_ijepa.transformer import  VisionTransformer
from simple_ijepa.ijepa import IJEPA
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from einops.layers.torch import Rearrange
from torchvision.utils import save_image
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from lars import LARS

class EmbeddingDataset(Dataset):
    def __init__(self, embs, labels, device=None):

        self.embs = torch.as_tensor(embs, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

        if device is not None:
            self.embs = self.embs.to(device)
            self.labels = self.labels.to(device)

    def __len__(self):
        return len(self.embs)

    def __getitem__(self, idx):
        return self.embs[idx], self.labels[idx]
    
def get_image_embs_labels(model, dataloader, device):
    embs, labels = [], []
    for _, (images, targets) in enumerate(dataloader):
        with torch.no_grad():
            images = images.to(device)
            out = model(images)
            features = out.cpu().detach()
            features = features.mean(dim=1)
            embs.extend(features.tolist())
            labels.extend(targets.cpu().detach().tolist())
    return np.array(embs), np.array(labels)

if __name__ == "__main__":

    # params
    device = "cuda:1"

    # model
    encoder = VisionTransformer(
    image_size=96,
    patch_size=8,
    dim=512,
    depth=6,
    heads=6,
    mlp_dim=1024,
    ).to(device)

    ijepa = IJEPA(encoder,
                hidden_emb_dim=512,
                patch_size=8,
                num_targets=1)
    
    state_dict = torch.load("models/embdim=512_epoch=300_patchsize=8/training_model.pth", map_location=device)
    ijepa.load_state_dict(state_dict)
    ijepa = ijepa.to(device)

    # dataset
    train_transform = transforms.Compose([
        transforms.Resize(96),
    #     transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.Resize(96),
        transforms.ToTensor()
    ])
    batch_size = 256
    train_data = CIFAR10(root="data", train=True, transform=train_transform, download=True)
    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_data = CIFAR10(root="data", train=False, transform=test_transform, download=True)
    # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    train_loader_ = DataLoader(train_data, batch_size=64, num_workers=2)
    test_loader_ = DataLoader(test_data, batch_size=64, num_workers=2)
    train_emb, train_label = get_image_embs_labels(ijepa.target_encoder, train_loader_, device)
    test_emb, test_label = get_image_embs_labels(ijepa.target_encoder, test_loader_, device)
    train_dataset = EmbeddingDataset(train_emb, train_label)
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=2)
    test_dataset = EmbeddingDataset(test_emb, test_label)
    
    # train classifier
    classifier = torch.nn.Linear(512, 10, bias=True).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 50
    optimizer = LARS(classifier.parameters(), eta=0.02, lr=0.01, momentum=0.9, weight_decay=5e-4, clip_lr=True, exclude_bias_n_norm=True)
    
    def evaluate(model, dataset, device):
        model.eval()
        x, y = dataset.embs.to(device), dataset.labels.to(device)
        with torch.no_grad():
            pred = model(x).argmax(dim=1)
        acc = (pred == y).float().mean().item()
        return acc
    
    for epoch in range(epochs):
        classifier.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = classifier(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        acc = evaluate(classifier, test_dataset, device)    
        print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f} Accuracy={acc}")