import numpy as np
import torch
from torch.utils.data import DataLoader

class Gym:
    def __init__(self, train_set:DataLoader, val_set: DataLoader, epochs: int, directory:str, device:torch.device, learning_rate = .01, weight_decay = .000001, momentum = 0.9) -> None:
        self.epochs = epochs
        self.directory = directory
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.train_set = train_set
        self.val_set = val_set
    
    def _get_optimizer(self, net:torch.nn.Module) -> torch.optim.SGD:
        optimizer = torch.optim.SGD(net.parameters(),
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay,
                               momentum=self.momentum)
        return optimizer
    
    def _get_loss(self) -> torch.nn.CrossEntropyLoss:
        loss = torch.nn.CrossEntropyLoss()
        return loss
    
    def _train(self, net: torch.nn.Module, optimizer, loss_function) -> None:
        net = net.train()
        net.half()
        for images, labels in self.train_set:
            images = images.to(self.device).half()
            labels = labels.to(self.device)
            outputs = net(images)

            loss = loss_function(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        net.float()
        
    def _test(self, net: torch.nn.Module, loss_function):
        samples = 0
        cumulative_loss = 0.0
        correct = 0
  
        net.eval()

        with torch.no_grad():
            for images, labels in self.val_set:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)

                loss = loss_function(outputs, labels)

                samples += images.size(dim=0)
                cumulative_loss += loss.item()

                _, predict = outputs.max(1)
                correct += predict.eq(labels).sum().item()

        return cumulative_loss, correct/samples * 100
    
    def workout(self, net: torch.nn.Module, load_checkpoint: bool = False) -> None:
        optimizer = self._get_optimizer(net=net)
        loss_function = self._get_loss()
        
        if load_checkpoint:
            checkpoint = torch.load(self.directory)
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss_value = checkpoint['loss']
            accuracy = checkpoint['accuracy']
            if epoch >= self.epochs:
                raise Exception("Model already trained for the desired number of epochs")
            print("------ Epoch {}/{} - Perofrmance on validation set (CHECKPOINT) ------".format(epoch + 1, self.epochs))
            print("Loss function value: {:.2f} \t Accuracy: {:.2f}%\n".format(loss_value, accuracy))
            
            
        else:
            print("------ INITIAL PERFORMANCE ON TEST SET ------")
            loss_value, accuracy = self._test(net, loss_function=loss_function)
            print("Loss function value: {:.2f} \t Accuracy: {:.2f}%\n".format(loss_value, accuracy))
            epoch = 0
        
        while epoch < self.epochs:
            self._train(net, optimizer, loss_function)
            loss_value, accuracy = self._test(net, loss_function=loss_function)
            print("------ Epoch {}/{} - Perofrmance on validation set ------".format(epoch + 1, self.epochs))
            print("Loss function value: {:.2f} \t Accuracy: {:.2f}%\n".format(loss_value, accuracy))
            epoch += 1
            torch.save({
                'epoch': epoch - 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_value,
                'accuracy': accuracy
            }, self.directory)
        
        print("\n------ PERFORMANCE ON TEST SET AFTER TRAINING ------")
        loss_value, accuracy = self._test(net, loss_function=loss_function)
        print("Loss function value: {:.2f} \t Accuracy: {:.2f}%\n".format(loss_value, accuracy))
        
        