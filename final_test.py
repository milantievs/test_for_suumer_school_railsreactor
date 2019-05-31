import argparse
import os
import numpy as np
from PIL import Image, ImageFilter

def make_hash(name1):
    img1 = np.array(np.asarray(Image.open(f'{name1}').resize((32, 32), Image.ANTIALIAS).convert('L')), dtype=np.double)
    dct_matrix = np.zeros((32, 32))
    for i in range(32):
        for j in range(32):
            ij = 0
            for n1 in range(32):
                temp = 0
                for n2 in range(32):
                    temp += img1[n1][n2] * np.cos(np.pi * (n1 + 0.5) * i / 32) * np.cos(np.pi * (n2 + 0.5) * j / 32)
                ij += temp
            dct_matrix[i][j] = ij

    res = dct_matrix[:8, :8]
    res_c = np.copy(res)
    res_c[0][0] = 0
    mean_res = res_c.mean()

    return (np.where(res < mean_res, 0, 1)).reshape((64,))


def hamm(a, b):
    return np.count_nonzero(a != b)


def pHash(path='./dev_dataset'):
    list_of_images = os.listdir(path)
    k = len(list_of_images)
    res = np.zeros((k, k))
    hash_of_images = []

    for i in list_of_images:
        hash_of_images.append(make_hash(path + '/' + i))

    for i in range(k):
        for j in range(i + 1, k):
            res[i][j] = hamm(hash_of_images[i], hash_of_images[j])

    return hash_of_images


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path')
    args = parser.parse_args()
    path = args.path
    phash_of_images = pHash(path)

    f = 0
    list_of_images = os.listdir(path)
    k = len(list_of_images)
    for i in range(k):
        for j in range(i + 1, k):
            if hamm(phash_of_images[i], phash_of_images[j]) < 21:
                print(list_of_images[i], list_of_images[j])

