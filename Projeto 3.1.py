#imports
import os
import numpy as np
import pickle
import shutil

#from imports
# for loading/processing the images  
from keras.preprocessing.image import load_img 
from keras.applications.vgg16 import preprocess_input
# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model
# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# Download images
from simple_image_download import simple_image_download as SID

def extract_features(file, model):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224,224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img) 
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3) 
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features
def save_cluster(groups, cluster, path, extension='.jpg',database='srcs'):
    # gets the list of filenames for a cluster
    save_path = os.path.join(path, 'clusters')
    cluster_path = os.path.join(save_path,str(cluster))
    files = groups

    if not (os.path.exists(save_path)):
        os.mkdir(save_path)

    if not os.path.exists(cluster_path):
    
        os.mkdir(cluster_path)


    
    nfiles = len([name for name in os.listdir(cluster_path) if os.path.isfile(os.path.join(cluster_path,name))])
    for index,file in enumerate(files):
        index += nfiles
        img = os.path.join(path,database,file)
        saving = os.path.join(save_path,str(cluster),str(cluster).zfill(2) + "_" + str(index).zfill(4) + extension)  
        shutil.copyfile(img,saving)   
def clustering(src_name, quantity, main_directory= 'database', source= 'srcs', extension='.jpg'):
    data_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(data_path,main_directory,src_name)
    path = os.path.join(data_path,source)
    os.chdir(path)

    # this list holds all the image filename
    images = []
    # creates a ScandirIterator aliased as files
    with os.scandir(path) as files:
    # loops through each file in the directory
        for file in files:
            if file.name.endswith('.jpeg'):
                # adds only the image files to the flowers list
                images.append(file.name)
            elif file.name.endswith('.jpg'):
                # adds only the image files to the flowers list
                images.append(file.name)  
            elif file.name.endswith('.png'):
                # adds only the image files to the flowers list
                images.append(file.name)  
                
    model = VGG16()
    model = Model(inputs = model.inputs, outputs = model.layers[-2].output)
    
    data = {}
    # lop through each image in the dataset
    for image in images:
        # try to extract the features and update the dictionary
        try:
            feat = extract_features(image,model)
            data[image] = feat
        # if something fails, save the extracted features as a pickle file (optional)
        except:
            with open(data_path,'wb') as file:
                pickle.dump(data,file)
            
    # get a list of the filenames
    filenames = np.array(list(data.keys()))
    # get a list of just the features
    feat = np.array(list(data.values()))
    # reshape so that there are 210 samples of 4096 vectors
    feat = feat.reshape(-1,4096)

    if(len(images) >= 100):
        # reduce the amount of dimensions in the feature vector
        pca = PCA(n_components=100, random_state=22)
        pca.fit(feat)
        x = pca.transform(feat)
    else:
        x = feat

    # cluster feature vectors
    kmeans = KMeans(n_clusters=quantity, random_state=22)
    kmeans.fit(x)

    # holds the cluster id and the images { id: [images] }
    groups = {}
    for file, cluster in zip(filenames,kmeans.labels_):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)
    
    for i in range(len(groups)):
        save_cluster(groups[i], i, data_path, extension= extension)

def main():
  print("Insira um termo para ser usado para buscar imagens no Google.")
  term = input()
  print("Quantas imagens devem ser buscadas?")
  qut = int(input())

  response = SID.simple_image_download
  response().download(term, qut, extensions=('.jpeg','.jpg','.png'), main_directory=('database/'+term))

  clust = int(input("Deseja agrupar em quantos grupos? "))
  clustering(term, clust, extension='.png')
    
if __name__== "__main__" :  
  main()