# file to take an image and a semgmentation mask of zeros and ones and cut out the card part from the image into a horizontal angle

from utils import Transformations

import numpy as np
import cv2


def rotate_image(image, angle):
    """
    Rotate the image by the specified angle.
    """
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return rotated_image


def crop_card(image, mask):
    """
    Crop the image such that the card occupies the entire width.
    """
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(contours[0])

    # Calculate the angle of rotation needed to make the card horizontal
    angle = np.arctan2(h, w) * (180 / np.pi)

    # Rotate the image and mask
    rotated_image = rotate_image(image, angle)
    rotated_mask = rotate_image(mask, angle)

    # Crop the rotated image
    cropped_image = rotated_image[y:y + h, x:x + w]

    return cropped_image


# Example usage
# Assuming you have 'image' and 'mask' numpy arrays containing your image and mask segmentation respectively
# Replace 'image' and 'mask' with your actual data

# Call the function to crop the card


image_path = "C:/Users/alex/Downloads/WhatsApp Image 2024-03-30 at 07.02.01_30d53384.jpg"
mask_path = "outputresized.png"

# read both image and mask
image = cv2.imread(image_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

cropped_card = crop_card(image, mask)
cv2.imwrite("outputwrapedmine.jpg", cropped_card)

# write image
result = Transformations.convert_object(mask, image)

cv2.imwrite("outputwrapped.jpg", result)
