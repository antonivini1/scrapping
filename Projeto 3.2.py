import os
import cv2
import shutil
from simple_image_download import simple_image_download as SID

def howSimilar(file_name, target_name, src_name=None, main_directory= 'database/'):
    if(src_name != None):
        data_path = os.path.join(main_directory,src_name)
        path = os.path.join(data_path,"srcs")


    if(os.path.isfile(os.path.join(main_directory, file_name))):
        file = os.path.join(main_directory, file_name)
    elif(os.path.isfile(file_name)):
        file = file_name
    else:
        file = os.path.join(path, file_name)


    if(os.path.isfile(os.path.join(main_directory, target_name))):
        target = os.path.join(main_directory, target_name)
    elif(os.path.isfile(target_name)):
        target = target_name
    else:
        target = os.path.join(path, target_name)
    
    
    original = cv2.imread(file)
    image_to_compare = cv2.imread(target)

    #1) Check if 2 images are equals
    if original.shape == image_to_compare.shape:
        #print("The images have same size and channels")
        difference = cv2.subtract(original, image_to_compare)
        b, g, r = cv2.split(difference)

        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            #print("The images are completely Equal")
            return float(100)
        # else:
        #     print("The images are NOT equal")
    
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
def moreSimilar(file_name, clst_name, main_directory= "database/"):
    data_path = os.path.join(main_directory, clst_name)
    cluster_path = os.path.join(data_path, 'clusters')

    titles = []
    pathes = []
    all_similarities = []
    for subdirectory in os.scandir(cluster_path):
        if(subdirectory.is_dir()):
            cluster = subdirectory.path
            for file in os.listdir(cluster):
                target = os.path.join(cluster,file)
                titles.append(file)
                pathes.append(os.path.abspath(cluster))
                all_similarities.append(howSimilar(file_name, target,src_name= clst_name))
    

    closest = max(all_similarities)
    count = 0
    for index in all_similarities:
        if(index == closest):
            title = titles[count]
            pathr = pathes[count]
        else:
            count += 1
    
    return closest, title, pathr

def main(main_directory='database/',source= 'srcs'):
    print("Insira um termo para ser usado para buscar imagens no Google.")
    term = input()
    print("Quantas imagens devem ser buscadas?")
    qut = int(input())

    #response = SID.simple_image_download
    #response().download(term, qut, extensions=('.jpeg','.jpg','.png'), main_directory=(main_directory+term))

    data_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(data_path,main_directory,term)
    path = os.path.join(data_path,source)
    for file in os.listdir(path):
        semelhança,nome,dest = moreSimilar(file, "Google")
        orig = os.path.join(path,file)
        dest = os.path.join(dest,file)
        shutil.copyfile(orig,dest)

        

if __name__== "__main__" :  
  main()

