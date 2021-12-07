# imports
import os
import cv2
import numpy as np
import pickle
import shutil
import requests


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
# for web scrapping
from tqdm import tqdm
from time import perf_counter, sleep
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from simple_image_download import simple_image_download as sid
# for comparision
from sewar import vifp, uqi


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
def save_cluster(groups, cluster, path, extension='.jpg', save= 'clusters'):
    # gets the list of filenames for a cluster
    save_path = os.path.join(path, save)
    cluster_path = os.path.join(save_path,str(cluster))
    files = groups

    if not (os.path.exists(save_path)):
        os.mkdir(save_path)

    if not (os.path.exists(cluster_path)):
        os.mkdir(cluster_path)


    
    nfiles = len([name for name in os.listdir(cluster_path) if os.path.isfile(os.path.join(cluster_path,name))])
    for index,file in enumerate(files):
        index += nfiles
        img = os.path.join(path,file)
        saving = os.path.join(save_path,str(cluster),str(cluster).zfill(2) + "_" + str(index).zfill(4) + extension)  
        shutil.copyfile(img,saving)
def clustering(src_name, quantity):
    data_path = os.getcwd()
    path = os.path.join(data_path,src_name)
    os.chdir(path)

    # this list holds all the image filename
    images = []
    # creates a ScandirIterator aliased as files
    with os.scandir(path) as files:
    # loops through each file in the directory
        for file in files:
            if file.name.endswith('.jpeg'):
                images.append(file.name)
            elif file.name.endswith('.jpg'):
                images.append(file.name)  
            elif file.name.endswith('.png'):
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
            with open(path,'wb') as file:
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
        save_cluster(groups[i], i, path)
    os.chdir(data_path)

def howSimilar(file_name, target_name):
    data_path = os.getcwd()
    file = os.path.join(data_path,file_name)
    target = os.path.join(data_path,target_name)

    original = cv2.imread(file)
    image_to_compare = cv2.imread(target)


    # 1) Check if 2 images are equals
    if original.shape == image_to_compare.shape:
        difference = cv2.subtract(original, image_to_compare)
        b, g, r = cv2.split(difference)

        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            return float(100)
    

    # 2) Check for similarities between the 2 images
    sift = cv2.xfeatures2d.SIFT_create()
    kp_1, desc_1 = sift.detectAndCompute(original, None)
    kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc_1, desc_2, k=2)

    good_points = []
    for m, n in matches:
        if m.distance < 0.6*n.distance:
            good_points.append(m)

    # Define how similar they are
    number_keypoints = 0
    if len(kp_1) <= len(kp_2):
        number_keypoints = len(kp_1)
    else:
        number_keypoints = len(kp_2)
    percentage_similarity = len(good_points) / number_keypoints * 100
    return percentage_similarity
def moreSimilar(file_name, clst_name):
    data_path = os.getcwd()
    img_path = os.path.join(data_path, file_name)


    titles = []
    pathes = []
    all_similarities = []
    for sub in os.scandir(clst_name):
        if(sub.is_dir()):
            for file in os.listdir(sub.path):
                img2comp = os.path.join(sub.path, file)
                if(os.path.isfile(img2comp)):
                    if(file.endswith('.jpeg') or file.endswith('.jpg') or file.endswith('.png')):
                        titles.append(file)
                        pathes.append(sub.path)
                        all_similarities.append(howSimilar(img_path, img2comp))

    
    closest = max(all_similarities)
    count = 0
    for index in all_similarities:
        if(index == closest):
            title = titles[count]
            pathr = pathes[count]
        else:
            count += 1
    
    return closest, title, pathr
def mov_files(path,target):
    for file in os.listdir(path):
        file_path = os.path.join(path,file)
        if(os.path.isfile(file_path)):
            extension = file.endswith('.jpeg') or file.endswith('.jpg') or file.endswith('.png')
            if(extension):
                semelhança,nome,dest = moreSimilar(file_path, target)
                target_path = os.path.join(dest, file)
                shutil.copyfile(file_path,target_path)
                index = len([name for name in os.listdir(dest) if (os.path.isfile(os.path.join(dest,name)) and extension)])
                extension = os.path.splitext(file)
                new_name = rename(dest, index, extension[1])
                os.rename(target_path,new_name)
    



def insert_url():
    print("\nInsira um URL:")
    url = input("Link- ")
    try:
        response = requests.get(url)
        print("URL valido.")
        return url
    except requests.ConnectionError as exception:
        print("URL invalido.")
        insert_url()
def check_path(url=None):
    print("\nInsira uma pasta para salvar as imagens.")
    path = input("Pasta- ")
    
    if(url != None):
        if not path:
            # if path isn't specified, use the domain name of that url as the folder name
            path = urlparse(url).netloc
    if not (os.path.exists(path)):
        os.mkdir(path)
    return path
def is_valid(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)
def get_all_images(url):
    soup = BeautifulSoup(requests.get(url).content, "html.parser")
    urls = []
    for img in tqdm(soup.find_all("img"), "Extracting images"):
        img_url = img.attrs.get("src")
        if not img_url:
            continue
        img_url = urljoin(url, img_url)
        try:
            pos = img_url.index("?")
            img_url = img_url[:pos]
        except ValueError:
            pass
        if is_valid(img_url):
            urls.append(img_url)
    return urls
def download(url, pathname):
    response = requests.get(url, stream=True)
    file_size = int(response.headers.get("Content-Length", 0))
    filename = os.path.join(pathname, url.split("/")[-1])
    extension = os.path.splitext(filename)

    if(filename.find('.svg') != -1 or extension[1] == ''):
        return
    elif(extension[1] == '.jpeg' or extension[1] == '.jpg' or extension[1] == '.png'):
        progress = tqdm(response.iter_content(1024), f"Downloading {filename}", total=file_size, unit="B", unit_scale=True, unit_divisor=1024)
        with open(filename, "wb") as f:
            for data in progress.iterable:
                f.write(data)
                progress.update(len(data))
    else:
        return

def rename(cluster, index, extension):
    new_name = os.path.normpath(cluster)
    new_name = str(os.path.basename(new_name))
    new_name = new_name.zfill(2) + "_" + str(index).zfill(4) + extension
    new_name = os.path.join(cluster,new_name)
    return new_name
def delet_files(path):
    print("\n\n\n\n\n")
    while True:
        print("Deletar imagens de origem?")
        print("[S] Sim")
        print("[N] Não")
        opt2 = input("R- ")
        if(opt2 == 'S' or opt2 == 's'):
            for file in os.listdir(path):
                if(os.path.isfile(os.path.join(path,file))):
                    if(file.endswith('.jpeg') or file.endswith('.jpg') or file.endswith('.png')):
                        os.remove(os.path.join(path,file))
            return False
            break
        elif(opt2 == 'N' or opt2 == 'n'):
            return True
        else:
            print("Opção Invalida.")
def Menu(path):
    while True:
        print("\n\nDeseja agrupar as imagens por similaridade ou inserir a uma database ja existente? ")
        print("[1] Agrupar por similaridade.")
        print("[2] Adicionar a uma database.")
        print("[3] Voltar ao menu.")
        print("[4] Finalizar o programa.")

        try:
            opt = int(input("\nR- "))

            if(opt == 1):
                print("\nDeseja agrupar em quantos grupos?")
                qut = int(input("R- "))
                clustering(path,qut)
                end = delet_files(path)
            elif(opt == 2):
                print("Em qual database deseja adicionar?")
                npath = input("R- ")
                mov_files(path, npath)
                end = delet_files(path)
            elif(opt == 3):
                return False
            elif(opt == 4):
                return True
            else:
                print("Opção invalida.")
            
            if not (end):
                return False
        except ValueError:
            print("Aqui")
            print("Escolha um valor numerico.")
def Webscrapping():
    url = insert_url()
    path = check_path(url)
 

    imgs = get_all_images(url)
    for img in imgs:
        download(img, path)

    
    opt = Menu(path)
    return opt
def Googlesearch():
    print("\nInsira um termo para ser usado para buscar imagens no Google.")
    term = input("Termo- ")
    print("\nQuantas imagens devem ser buscadas?")
    qut = int(input("Quantidade- "))
    path = check_path()


    down = sid.simple_image_download()
    down.download(term, qut, extensions=('.jpeg','.jpg','.png'),main_directory=path,sub=False)


    opt = Menu(path)
    return opt


def main():
    print("Universidade Federal da Paraiba - CI")
    print("Processamento Digital de Imagem - 2021.1")
    print("Projeto Final - 06/12/2021")
    print("Tema: Webscrapping e agrupamento por similaridade")
    print("Nome: Antoni Vinicius da Silva Soares")
    print("Matricula: 20190031597")
    print("Nome: Filipe Trindade de Oliveira")
    print("Matricula: 20190019912")
    print("Professor: Leonardo Vidal Batista")
    
    print("\n\n")
    print("Esse programa tem como objetivo coletar e baixar imagens de uma pagina da internet, que o usuario forneça o link, ou então")
    print("coletar e baixar n imagens do Google Imagens de um certo termo, o usuario tambem deve fornecer os termos e quantidades.")
    
    while True:
        print("\n\n\nSelecione uma operação?\n")
        print("[1] Baixar imagens de uma página.")
        print("[2] Pesquisar um termo no Google Imagens.")
        print("[3] Comparar duas imagens.")
        print("[4] Comparar uma imagem com uma database.")
        print("[5] Finalizar o programa.")
        

        try:
            opt = int(input("\nR- "))
            if(opt == 1):
                end = Webscrapping()
            elif(opt == 2):
                end = Googlesearch()
            elif(opt == 3):
                print("\nInsira o caminho da primeira imagem.")
                img1 = input("R- ")
                print("\nInsira o caminho da segunda imagem.")
                img2 = input("R- ")
                pct = howSimilar(img1, img2)
    
                im1 = cv2.imread(img1)
                im2 = cv2.imread(img2)
                im2 = cv2.resize(im2,(im1.shape[1], im1.shape[0]))
                vifpv = vifp(im1,im2)
                uqiv = uqi(im1,im2)

                img1 = os.path.normpath(os.path.basename(img1))
                img2 = os.path.normpath(os.path.basename(img2))
                print("\n\n\n\n\n")
                print("A "+img1+" é",pct,"%"+" similar a "+img2)
                print("Pelo VIFP elas tem uma semelhança de",vifpv,"(1 é o melhor).")
                print("Pelo UQI elas tem uma semelhança de",uqiv,"(1 é o melhor).")

                sleep(3)
                end = False
            elif(opt == 4):
                print("\nInsira o caminho da primeira imagem.")
                img = input("R- ")
                print("\nInsira o caminho da database.")
                path = input("R- ")
                pct, nome, pathr = moreSimilar(img, path)

                im1 = cv2.imread(img)
                im2 = cv2.imread(os.path.join(pathr,nome))
                im2 = cv2.resize(im2,(im1.shape[1], im1.shape[0]))
                vifpv = vifp(im1,im2)
                uqiv = uqi(im1,im2) 

                img = os.path.normpath(os.path.basename(img))
                print("A imagem mais similar a "+img+" é "+nome+" com ",pct,"%"+" de similaridade.")
                print("Pelo VIFP elas tem uma semelhança de",vifpv,"(1 é o melhor).")
                print("Pelo UQI elas tem uma semelhança de",uqiv,"(1 é o melhor).")
                end = False
            elif(opt == 5):
                end = True
            else:
                print("\nOpção invalida\n")
            
            if(end):
                print("Finalizando o programa.")
                sleep(2)
                break
        except ValueError:
            print("Escolha um valor numerico.")

    
    


if __name__ == "__main__": 
    main()