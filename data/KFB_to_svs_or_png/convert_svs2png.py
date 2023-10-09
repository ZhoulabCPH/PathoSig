import openslide
import numpy as np
import PIL.Image
import os
def readsvs():
    svs_dir = './output/'
    # svs_path_list = []
    svs_name_list = []
    for i in os.listdir(svs_dir):
        path_name = os.path.join(i)
        # path_svs = os.path.join(svs_dir,i)
        svs_name_list.append(path_name)
    return svs_name_list 

def main(svs_name):

    svs_dir = 'output/'
    test = openslide.open_slide(svs_dir + svs_name)
    img = np.array(test.read_region((0, 0), 0, test.dimensions))
    output_path = r'output/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    png_name = svs_name.split('.')[0]
    PIL.Image.fromarray(img).save('./output/' + png_name + '.png')


if __name__ == '__main__':
    svs_name_list = readsvs()
    for i in range(len(svs_name_list)):
        abc = main(svs_name_list[i])




