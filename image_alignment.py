import cv2
import numpy as np
import matplotlib.pyplot as plt

def align_images(ref_filename, im_filename):
    # Read reference image
    print("Reading reference image:", ref_filename)
    im1 = cv2.imread(ref_filename, cv2.IMREAD_COLOR)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

    # Read image to be aligned
    print("Reading image to align:", im_filename)
    im2 = cv2.imread(im_filename, cv2.IMREAD_COLOR)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors
    MAX_NUM_FEATURES = 500
    orb = cv2.ORB_create(MAX_NUM_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

    # Match features
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches = sorted(matches, key=lambda x: x.distance)

    # Remove not so good matches
    num_good_matches = int(len(matches) * 0.1)
    matches = matches[:num_good_matches]

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    retval = cv2.findHomography(points2, points1, cv2.RANSAC)
    h, mask = retval[0], retval[1]

    # Use homography to warp image
    height, width, channels = im1.shape
    im2_reg = cv2.warpPerspective(im2, h, (width, height))

    # Display key features in another window
    im1_display = cv2.drawKeypoints(im1, keypoints1, outImage=np.array([]), color=(255, 0, 0),
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im2_display = cv2.drawKeypoints(im2, keypoints2, outImage=np.array([]), color=(255, 0, 0),
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.figure(figsize=[20, 10])
    plt.subplot(121); plt.imshow(im1_display); plt.axis('off'); plt.title("Reference Image Key Features")
    plt.subplot(122); plt.imshow(im2_display); plt.axis('off'); plt.title("Test Image Key Features")
    plt.show()

    # Display results
    plt.figure(figsize=[20, 10])

    plt.subplot(131); plt.imshow(im1); plt.axis('off'); plt.title("Reference Image")
    plt.subplot(132); plt.imshow(im2); plt.axis('off'); plt.title("Test Image")
    plt.subplot(133); plt.imshow(im2_reg); plt.axis('off'); plt.title("Aligned Image")

    plt.show()

    

if __name__ == "__main__":
    reference_filename = "reference_imgae.jpg"
    image_filename = "test_image.jpg"
    align_images(reference_filename, image_filename)
