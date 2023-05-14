import sys
import subprocess

def get_dataloaders(batch_size = 64, resize = 112):
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

    testset = pyvww.pytorch.VisualWakeWordsClassification(root="/content/all2014", #Folder containing all the images
                                                          annFile="/content/drive/MyDrive/annotations/instances_train.json", #Annotation file
                                                          transform = transforms #Transforms to be applied to the images
                                                          )

    valset = pyvww.pytorch.VisualWakeWordsClassification(root="/content/all2014", #Folder containing all the images
                                                          annFile="/content/drive/MyDrive/annotations/instances_val.json", #Annotation file
                                                          transform = transforms #Transforms to be applied to the images
                                                          )
    
    test_load = DataLoader(
        testset, batch_size=batch_size, shuffle=True
    )

    val_load = DataLoader(
        valset, batch_size=batch_size, shuffle=False
    )
    
    return testset, test_load, valset, val_load
    
    
    