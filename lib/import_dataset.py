import os
def unzip_dataset():
    os.system("cd /content/drive/MyDrive/")
    os.system("unzip -q all2014.zip -d /content")
    os.system("cd /content")