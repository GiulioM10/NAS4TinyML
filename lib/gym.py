# GM 05/17/23
import numpy as np
import torch
from torch.utils.data import DataLoader
from space import init_model, kaiming_normal
import matplotlib.pyplot as plt

class Gym:
    def __init__(self, train_set:DataLoader, val_set: DataLoader, epochs: int, directory:str, device:torch.device, learning_rate = .01, weight_decay = .00001, momentum = 0.9) -> None:
        """This object handles the training proceess and performance assesment of architectures

        Args:
            train_set (DataLoader): DataLoader containing the train images
            val_set (DataLoader): DataLoader containing the validation/test images
            epochs (int): The numbers of epochs for which we want to train our model
            directory (str): Path to the file in which to store checkpoints
            device (torch.device): Device to be used to perform computations
            learning_rate (float, optional): Learning rate of the optimizer. Defaults to .005.
            weight_decay (float, optional): Weight decay of the optimizer. Defaults to .005.
            momentum (float, optional): Momentum of the optimizer. Defaults to 0.9.
        """
        self.epochs = epochs
        self.directory = directory
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.train_set = train_set
        self.val_set = val_set
    
    def _get_optimizer(self, net:torch.nn.Module) -> torch.optim.SGD:
        """Return an optimizer to train a given architecture

        Args:
            net (torch.nn.Module): The architecture to be trained

        Returns:
            torch.optim.SGD: A Stochastic Gradient Descent optimizer with the previously specified hyper-parameters
        """
        optimizer = torch.optim.SGD(net.parameters(),
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay,
                               momentum=self.momentum)
        return optimizer
    
    def _get_loss(self) -> torch.nn.CrossEntropyLoss:
        """Returns a loss function to evaluate performance

        Returns:
            torch.nn.CrossEntropyLoss: A cross-entropy loss function
        """
        loss = torch.nn.CrossEntropyLoss()
        return loss
    
    def _get_lr_scheduler(self, optimizer):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.316)
        return scheduler
    
    def _train(self, net: torch.nn.Module, optimizer, loss_function) -> None:
        """Perform an epoch of training for a network

        Args:
            net (torch.nn.Module): The network to be trained
            optimizer (_type_): An optimizer
            loss_function (_type_): A loss function
        """
        samples = 0
        cumulative_loss = 0.0
        correct = 0
        
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
            
            samples += images.size(dim=0)
            cumulative_loss += loss.item()
            
            _, predict = outputs.max(1)
            correct += predict.eq(labels).sum().item()
        
        net.float()
        return cumulative_loss, correct/samples * 100
        
    def _test(self, net: torch.nn.Module, loss_function):
        """Get performance of a model on a test/validation set

        Args:
            net (torch.nn.Module): The model
            loss_function (_type_): Loss function used to evaluate performance

        Returns:
            tuple: Value of the loss and accuracy
        """
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
        """Train a model using the equipement of the gym

        Args:
            net (torch.nn.Module): The model to be trained
            load_checkpoint (bool, optional): Wether to load the weights from a previous workout session. Defaults to False.

        Raises:
            Exception: Model already trained for the requested number of epochs
        """
        optimizer = self._get_optimizer(net=net)
        loss_function = self._get_loss()
        scheduler = self._get_lr_scheduler(optimizer)
        
        if load_checkpoint:
            checkpoint = torch.load(self.directory, map_location = self.device)
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            epoch = checkpoint['epoch']
            loss_value = checkpoint['loss']
            accuracy = checkpoint['accuracy']
            results = checkpoint['results']
            if epoch >= self.epochs:
                raise Exception("Model already trained for the desired number of epochs")
            print("------ Epoch {}/{} - Perofrmance on validation set (CHECKPOINT) ------".format(epoch, self.epochs))
            print("Loss function value: {:.2f} \t Accuracy: {:.2f}%\n".format(loss_value, accuracy))
            
            
        else:
            results = {
                "train_loss": [],
                "train_accuracy": [],
                "val_loss": [],
                "val_accuracy": []
            }
            print("------ INITIAL PERFORMANCE ON VALIDATION SET ------")
            loss_value, accuracy = self._test(net, loss_function=loss_function)
            print("Loss function value: {:.2f} \t Accuracy: {:.2f}%\n".format(loss_value, accuracy))
            epoch = 0
        
        while epoch < self.epochs:
            train_loss, train_accuracy = self._train(net, optimizer, loss_function)
            loss_value, accuracy = self._test(net, loss_function=loss_function)
            print("------ Epoch {}/{} - Perofrmance on train set ------".format(epoch + 1, self.epochs))
            print("Loss function value: {:.2f} \t Accuracy: {:.2f}%\n".format(train_loss, train_accuracy))
            print("------ Epoch {}/{} - Perofrmance on validation set ------".format(epoch + 1, self.epochs))
            print("Loss function value: {:.2f} \t Accuracy: {:.2f}%\n".format(loss_value, accuracy))
            epoch += 1
            scheduler.step()
            results["val_loss"].append(loss_value)
            results["val_accuracy"].append(accuracy)
            results["train_loss"].append(train_loss)
            results["train_accuracy"].append(train_accuracy)
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss_value,
                'accuracy': accuracy,
                'results': results
            }, self.directory)
        
        print("\n------ PERFORMANCE ON TEST SET AFTER TRAINING ------")
        loss_value, accuracy = self._test(net, loss_function=loss_function)
        print("Loss function value: {:.2f} \t Accuracy: {:.2f}%\n".format(loss_value, accuracy))
        
    def compute_performance(self, net: torch.nn.Module, load_from_checkpoint: bool = True) -> tuple:
        """Get the performance metrics for an individual

        Args:
            net (torch.nn.Module): The model whose performance needs to be assesed
            load_from_checkpoint (bool, optional): Wether to load the weights from a previous workout session. Defaults to True.

        Returns:
            tuple: Loss value and accuracy of the model on the test/validation set
        """
        if load_from_checkpoint:
            checkpoint = torch.load(self.directory, map_location = self.device)
            net.load_state_dict(checkpoint['model_state_dict'])
        loss_function = self._get_loss()
        loss_value, accuracy = self._test(net, loss_function=loss_function)
        return loss_value, accuracy
    
    def show_performance(self, net: torch.nn.Module, load_from_checkpoint: bool = True) -> None:
        """Print performance info to screen

        Args:
            net (torch.nn.Module): The model
            load_from_checkpoint (bool, optional): Wether to load the weights from a previous workout session. Defaults to True.
        """
        loss_value, accuracy = self.compute_performance(net, load_from_checkpoint)
        print("\n------ PERFORMANCE ON TEST SET ------")
        print("Loss function value: {:.2f} \t Accuracy: {:.2f}%\n".format(loss_value, accuracy))
        
    def show_learning_curves(self):
        checkpoint = torch.load(self.directory, map_location = self.device)
        results = checkpoint['results']
        epoch = checkpoint['epoch']
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        ax1.plot(np.arange(1, epoch+1), results["train_loss"], label = "train")
        ax1.scatter(np.arange(1, epoch+1), results["val_loss"], label = "validation")
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss Function')
        ax1.title("Learning curves (Loss)")
        ax1.legend()
        ax2.plot(np.arange(1, epoch+1), results["train_accuracy"], label = "train")
        ax2.scatter(np.arange(1, epoch+1), results["val_accuracy"], label = "validation")
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.title("Learning curves (Accuracy)")
        ax2.legend()
        
        