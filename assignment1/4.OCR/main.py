import imageio.v2 as imageio
from im2segment import im2segment

if __name__ == "__main__":
    I = imageio.imread("im7.jpg")
    S = im2segment(I, show_steps=True, connectivity=8)
    print("Masks:", len(S))
