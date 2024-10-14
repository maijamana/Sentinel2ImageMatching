import cv2
import matplotlib.pyplot as plt

def SIFT_keypoint_detection(image_path, plot=True):
    # SIFT parameter settings
    nfeatures = 0  # Maximum number of keypoints (0 = no limit)
    nOctaveLayers = 4
    contrastThreshold = 0.04
    edgeThreshold = 10
    sigma = 1.4

    # Initialize SIFT with custom parameters
    sift = cv2.SIFT_create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(img, None)
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    num_keypoints = len(keypoints)
    print(f"Number of keypoints detected: {num_keypoints}")

    if plot:
        # Display an image with key points
        plt.figure(figsize=(10, 10))
        plt.imshow(img_with_keypoints, cmap='gray')
        plt.title(f"Keypoints Detected: {num_keypoints}")
        plt.axis('off')
        plt.show()

    return keypoints, descriptors



def BRISK_keypoint_detection(img, plot=True):
    # BRISK parameter settings
    thresh = 30
    octaves = 4
    pattern_scale = 1.0

    # Initialize BRISK with custom parameters
    brisk = cv2.BRISK_create(thresh=thresh, octaves=octaves, patternScale=pattern_scale)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Detect keypoints and compute descriptors
    keypoints, descriptors = brisk.detectAndCompute(img, None)
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    num_keypoints = len(keypoints)
    print(f"Number of keypoints detected: {num_keypoints}")

    if plot:
        # Display an image with key points
        plt.figure(figsize=(10, 10))
        plt.imshow(img_with_keypoints, cmap='gray')
        plt.title(f"Keypoints Detected: {num_keypoints}")
        plt.axis('off')
        plt.show()

    return img_with_keypoints

def ORB_keypoint_detection(img, plot=True):
    # ORB parameter settings
    nfeatures = 20000
    scaleFactor = 2.0
    nlevels = 4
    edgeThreshold = 31
    firstLevel = 0
    WTA_K = 2
    scoreType = cv2.ORB_HARRIS_SCORE
    patchSize = 31

    # Initialize ORB with custom parameters
    orb = cv2.ORB_create(nfeatures=nfeatures,
                        scaleFactor=scaleFactor,
                        nlevels=nlevels,
                        edgeThreshold=edgeThreshold,
                        firstLevel=firstLevel,
                        WTA_K=WTA_K,
                        scoreType=scoreType,
                        patchSize=patchSize)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(img, None)

    num_keypoints = len(keypoints)
    print(f"Number of keypoints detected: {num_keypoints}")

    if plot:
        # Display an image with key points
        plt.figure(figsize=(10, 10))
        plt.imshow(img_with_keypoints, cmap='gray')
        plt.title(f"Keypoints Detected: {num_keypoints}")
        plt.axis('off')
        plt.show()
    return keypoints, descriptors



def BF_matcher(img1_path, img2_path, keypoints_img1, descriptors_img1, keypoints_img2, descriptors_img2, plot=True):
    # Load images
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # Match descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_img1, descriptors_img2, k=2)

    # Need to draw only good matches, so we create a mask
    matchesMask = [[0, 0] for _ in range(len(matches))]

    # Apply Lowe's ratio test
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:  # Apply the ratio test
            matchesMask[i] = [1, 0]  # Mark this match as good

    # Draw matches
    draw_params = dict(matchColor=(0, 255, 0),  # Green for good matches
                       singlePointColor=(255, 0, 0),  # Red for single points
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)

    img3 = cv2.drawMatchesKnn(img1, keypoints_img1, img2, keypoints_img2, matches, None, **draw_params)

    # Display images with matches  
    plt.imshow(img3)
    plt.title("BF-based Matches")
    plt.axis('off')
    plt.show()

    return img3


def FLANN_matcher(img1_path, img2_path, keypoints_img1, descriptors_img1, keypoints_img2, descriptors_img2, plot=True):
    # Параметри для FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # Ініціалізуємо FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # Match descriptors
    matches = flann.knnMatch(descriptors_img1, descriptors_img2, k=2)

    # Need to draw only good matches, so we create a mask
    matchesMask = [[0, 0] for _ in range(len(matches))]

    # Apply Lowe's ratio test
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:  # Apply the ratio test
            matchesMask[i] = [1, 0]  # Mark this match as good

    # Draw matches
    draw_params = dict(matchColor=(0, 255, 0),  # Green for good matches
                       singlePointColor=(255, 0, 0),  # Red for single points
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)

    img3 = cv2.drawMatchesKnn(img1, keypoints_img1, img2, keypoints_img2, matches, None, **draw_params)

    # Display images with matches  
    plt.imshow(img3)
    plt.title("FLANN-based Matches")
    plt.axis('off')
    plt.show()

    return img3


def RoMa_matcher(img1_path, img2_path, plot=True):
    device = 'cpu'
    matcher = get_matcher('tiny-roma', device=device)
    img_size = 1024

    img0 = matcher.load_image(img1_path, resize=img_size)
    img1 = matcher.load_image(img2_path, resize=img_size)

    # Match descriptors
    result = matcher(img0, img1)

    img3 = plot_matches(img0, img1, result)

    # Display images with matches 
    plot_matches(img0, img1, result, save_path='/content/matches_roma.png')
    return img3


def superglue_matcher(img1_path, img2_path, plot=True):
    device = 'cpu'
    matcher = get_matcher('superglue', device=device)
    img_size = 1024

    img0 = matcher.load_image(img1_path, resize=img_size)
    img1 = matcher.load_image(img2_path, resize=img_size)

    # Match descriptors
    result = matcher(img0, img1)

    img3 = plot_matches(img0, img1, result)

    # Display images with matches 
    plot_matches(img0, img1, result, save_path='/content/matches_superglue.png')
    return img3

def display_image(image_path):
    # Load images
    img1 = cv2.imread(image_path1)

    # Convert images to RGB for display
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    # Display images side by side
    ax1.imshow(img1)
    ax1.set_title('Image')
    ax1.axis('off')

    plt.show()
