import os
import cv2

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
        print("The images have same size and channels")
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

    # print("Keypoints 1ST Image: " + str(len(kp_1)))
    # print("Keypoints 2ND Image: " + str(len(kp_2)))
    # print("GOOD Matches:", len(good_points))
    # print("How good it's the match: ", len(good_points) / number_keypoints * 100)

    # result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)

    # cv2.imshow("result", cv2.resize(result, None, fx=0.4, fy=0.4))
    # cv2.imwrite("feature_matching.jpg", result)

    # cv2.imshow("Original", cv2.resize(original, None, fx=0.4, fy=0.4))
    # cv2.imshow("Duplicate", cv2.resize(image_to_compare, None, fx=0.4, fy=0.4))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
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

moreSimilar("Google_10.jpeg", "Google")
