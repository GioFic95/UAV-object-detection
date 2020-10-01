import string
import numpy as np
import cv2

chars = string.digits + string.ascii_uppercase + string.ascii_lowercase


if __name__ == '__main__':
    print(len(chars), chars)
    c = 30
    print(c, chars[c-1])
