import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import random


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    image_path = os.path.join(os.getcwd(), 'labels')
    xml_df = xml_to_csv(image_path)
    masks = [True] * int(round(len(xml_df) * 0.8, 0)) + [False] * int(round(len(xml_df) * 0.2, 0))
    random.shuffle(masks)
    train_mask = masks
    test_mask = np.invert(masks)
    train = xml_df[train_mask]
    test = xml_df[test_mask]
    train.to_csv('data/train_labels.csv', index=None)
    test.to_csv('data/test_labels.csv', index=None)
    print('Test cases: %d' % len(test))
    print('Successfully converted xml to csv.')


if __name__ == '__main__':
    main()
