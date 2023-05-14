import sys
import subprocess
import torch
import pyvww
import torchvision
from torchvision import transforms as T
from torch.utils.data import DataLoader

def get_dataloaders():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyvww'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'fvcore'])
    
    import sys
    import subprocess
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
        T.Resize((112, 112)), #Fixed size
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
    image, target = testset[5] #Extract an image from the dataset
    print(image.size())
    import numpy as np
    import matplotlib.pyplot as plt

    #Show image after retransforming it back
    img = T.ToPILImage()(T.Normalize(mean =[-0.4914, -0.4822, -0.4465], std = [1, 1, 1])(T.Normalize(mean = [0,0,0], std = [1/0.2023, 1/0.1994, 1/0.2010])(image)))
    plt.show(img)
    
    
    