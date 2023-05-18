# GM 05/17/23
import sys
import subprocess

def get_dataloaders(batch_size = 64, resize = 112, kaggle: bool = False):
    """Install the PYVWW and FVCORE Packages. after that build the dataloaders for the visual-wakewords dataset

    Args:
        batch_size (int, optional): Size of each mini-batch. Defaults to 64.
        resize (int, optional): The dimensions of the resized image (Non greater than 224). Defaults to 112.

    Returns:
        List: Dataloaders and datasets for visual-wakewords
    """
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyvww'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'fvcore'])
    
    import torch
    import pyvww
    import torchvision
    from torchvision import transforms as T
    from torch.utils.data import DataLoader
    
    norm = T.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )

    #Prepare the transforms applied to the images
    transforms = T.Compose([
        T.Resize((resize, resize)), #Fixed size
        T.ToTensor(), #Transform in a torch.tensor object
        norm #Normalization
    ])
    
    if kaggle:
        root = "/kaggle/input/visual-wake-words-224x224/all2014"
        annFile = "/kaggle/input/visual-wake-words-224x224/annotations/"
    else:
        root = "/content/all2014"
        annFile = "/content/drive/MyDrive/annotations/"
        
        

    testset = pyvww.pytorch.VisualWakeWordsClassification(root=root, #Folder containing all the images
                                                          annFile=(annFile + "instances_train.json"), #Annotation file
                                                          transform = transforms #Transforms to be applied to the images
                                                          )

    valset = pyvww.pytorch.VisualWakeWordsClassification(root=root, #Folder containing all the images
                                                          annFile=(annFile + "instances_val.json"), #Annotation file
                                                          transform = transforms #Transforms to be applied to the images
                                                          )
    
    test_load = DataLoader(
        testset, batch_size=batch_size, shuffle=True
    )

    val_load = DataLoader(
        valset, batch_size=batch_size, shuffle=False
    )
    
    return testset, test_load, valset, val_load
    
    
    