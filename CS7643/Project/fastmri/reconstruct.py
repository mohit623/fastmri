import os
import matplotlib.pyplot as plt


def reconstruct(fname, slice_num, image, output):
    # Create filename
    fname1 = ''.join(fname).replace('.', '_')
    # print(fname1)
    if not os.path.exists(fname1):
        os.makedirs(fname1)

    # Save input image
    title = '{}_{}_image'.format(fname1, slice_num)
    # title = fname1 + "_" + str(slice_num) + "_image"
    # print(title)
    
    # https://discuss.pytorch.org/t/convert-image-tensor-to-numpy-image-array/22887
    plt.imshow(image.permute(1, 2, 0).detach().cpu(), cmap='gray')
    plt.savefig(fname1 + "/{}.png".format(title))

    # Save output image
    title = '{}_{}_output'.format(fname1, slice_num)
    # title = fname1 + "_" + str(slice_num) + "_output"
    # print(title)

    plt.imshow(output.permute(1, 2, 0).detach().cpu(), cmap='gray')
    plt.savefig(fname1 + "/{}.png".format(title))
