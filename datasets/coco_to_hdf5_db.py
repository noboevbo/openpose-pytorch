#!/usr/bin/env python

"""
    Base src code: https://github.com/anatolix/keras_Realtime_Multi-Person_Pose_Estimation
"""

from pycocotools.coco import COCO
from scipy.spatial.distance import cdist
import numpy as np
import cv2
import os
import os.path
import h5py
import json

from config import cfg


def get_masks(img_dir, img_id, img_anns, coco):
    """
    Creates image masks for people in the image. mask_all contains all masked people, mask_miss contains
    only the masks for people without keypoints.
    """
    img_path = os.path.join(img_dir, "%012d.jpg" % img_id)
    img = cv2.imread(img_path)
    h, w, c = img.shape

    mask_all = np.zeros((h, w), dtype=np.uint8)
    mask_miss = np.zeros((h, w), dtype=np.uint8)

    flag = 0
    for p in img_anns:
        if p["iscrowd"] == 1:
            mask_crowd = coco.annToMask(p)
            temp = np.bitwise_and(mask_all, mask_crowd)
            mask_crowd = mask_crowd - temp
            flag += 1
            continue
        else:
            mask = coco.annToMask(p)

        mask_all = np.bitwise_or(mask, mask_all)
        if p["num_keypoints"] <= 0:
            mask_miss = np.bitwise_or(mask, mask_miss)

    if flag < 1:
        mask_miss = np.logical_not(mask_miss)
    elif flag == 1:
        mask_miss = np.logical_not(np.bitwise_or(mask_miss, mask_crowd))
        mask_all = np.bitwise_or(mask_all, mask_crowd)
    else:
        raise Exception("crowd segments > 1")

    mask_miss = mask_miss.astype(np.uint8)
    mask_miss *= 255

    mask_all = mask_all.astype(np.uint8)
    mask_all *= 255

    return img, mask_miss, mask_all


def get_persons(img_annotations):
    all_persons = []
    for person in img_annotations:
        person_dict = dict()

        person_center = [person["bbox"][0] + person["bbox"][2] / 2,
                         person["bbox"][1] + person["bbox"][3] / 2]

        person_dict["objpos"] = person_center
        person_dict["bbox"] = person["bbox"]
        person_dict["segment_area"] = person["area"]
        person_dict["num_keypoints"] = person["num_keypoints"]

        anno = person["keypoints"]

        person_dict["joint"] = np.zeros((17, 3))
        for part in range(17):
            person_dict["joint"][part, 0] = anno[part * 3]
            person_dict["joint"][part, 1] = anno[part * 3 + 1]

            if anno[part * 3 + 2] == 2:
                person_dict["joint"][part, 2] = 2
            elif anno[part * 3 + 2] == 1:
                person_dict["joint"][part, 2] = 1
            else:
                person_dict["joint"][part, 2] = 0

        # Scale provided -> Person Height / Img Height
        person_dict["scale_provided"] = person["bbox"][3] / cfg.general.input_height

        all_persons.append(person_dict)
    return all_persons


def get_main_persons(all_persons):
    """
    Returns only persons which:
        - Have enough joints
        - Have a great enough segmented area
        - Have a high enough distance to existing persons
    """
    prev_center = []
    main_persons = []
    for person in all_persons:

        # skip this person if parts number is too low or if
        # segmentation area is too small
        if person["num_keypoints"] < 5 or person["segment_area"] < 32 * 32:
            continue

        person_center = person["objpos"]

        # skip this person if the distance to exiting person is too small
        flag = 0
        for pc in prev_center:
            a = np.expand_dims(pc[:2], axis=0)
            b = np.expand_dims(person_center, axis=0)
            dist = cdist(a, b)[0]
            if dist < pc[2] * 0.3:
                flag = 1
                continue

        if flag == 1:
            continue

        main_persons.append(person)
        prev_center.append(np.append(person_center, max(person["bbox"][2], person["bbox"][3])))
    return main_persons


def get_annotation_template(img_id, img_index, img_rec, dataset_type):
    template = dict()
    template["dataset"] = dataset_type
    template["is_validation"] = img_index < cfg.dataset.val_size and 'val' in dataset_type
    template["img_width"] = img_rec['width']
    template["img_height"] = img_rec['height']
    template["image_id"] = img_id
    template["annolist_index"] = img_index
    template["img_path"] = '%012d.jpg' % img_id
    return template


def process_image(img_id, img_index, img_rec, img_annotations, dataset_type):
    print("Process image ID: ", img_id)

    all_persons = get_persons(img_annotations)
    main_persons = get_main_persons(all_persons)

    template = get_annotation_template(img_id, img_index, img_rec, dataset_type)

    for main_person in main_persons:

        instance = template.copy()

        instance["objpos"] = [main_person["objpos"]]
        # Joint Format: For each object, ground truth keypoints have the form [x1,y1,v1,...,xk,yk,vk], where x,y are the
        # keypoint locations and v is a visibility flag defined as v=0: not labeled, v=1: labeled but not visible,
        # and v=2: labeled and visible.
        instance["joints"] = [main_person["joint"].tolist()]
        instance["scale_provided"] = [main_person["scale_provided"]]

        lenOthers = 0

        for ot, operson in enumerate(all_persons):

            if main_person is operson:
                assert not "people_index" in instance, "several main persons? couldn't be"
                instance["people_index"] = ot
                continue

            if operson["num_keypoints"] == 0:
                continue

            instance["joints"].append(all_persons[ot]["joint"].tolist())
            instance["scale_provided"].append(all_persons[ot]["scale_provided"])
            instance["objpos"].append(all_persons[ot]["objpos"])

            lenOthers += 1

        assert "people_index" in instance, "No main person index"
        instance['num_other_people'] = lenOthers

        yield instance


def write_img(grp, img_grp, data, img, mask_miss, count, image_id, mask_grp):
    serializable_meta = data
    serializable_meta['count'] = count

    num_other_people = data['num_other_people']

    assert len(serializable_meta['joints']) == 1 + num_other_people, [len(serializable_meta['joints']), 1 + num_other_people]
    assert len(serializable_meta['scale_provided']) == 1 + num_other_people, [len(serializable_meta['scale_provided']), 1 + num_other_people]
    assert len(serializable_meta['objpos']) == 1 + num_other_people, [len(serializable_meta['objpos']), 1 + num_other_people]

    img_key = "%012d" % image_id
    if not img_key in img_grp:
        _, img_bin = cv2.imencode(".jpg", img)
        _, img_mask = cv2.imencode(".png", mask_miss)
        img_ds1 = img_grp.create_dataset(img_key, data=img_bin, chunks=None)
        img_ds2 = mask_grp.create_dataset(img_key, data=img_mask, chunks=None)

    key = '%07d' % count
    required = {'image': img_key, 'joints': serializable_meta['joints'], 'objpos': serializable_meta['objpos'],
                'scale_provided': serializable_meta['scale_provided']}
    ds = grp.create_dataset(key, data=json.dumps(required), chunks=None)
    ds.attrs['meta'] = json.dumps(serializable_meta)

    print('Writing sample %d' % count)


def process():
    datasets = [
        {
            'annotation_dir': cfg.dataset.val_annotation_dir,
            'image_dir': cfg.dataset.val_img_dir,
            'type': 'coco_val',
        },
        {
            'annotation_dir': cfg.dataset.train_annotation_dir,
            'image_dir': cfg.dataset.train_img_dir,
            'type': 'coco',
        },
    ]

    train_h5 = h5py.File(cfg.dataset.train_convert_hdf5, 'w')
    train_group = train_h5.create_group("dataset")
    train_write_count = 0
    train_grp_img = train_h5.create_group("images")
    train_grp_miss_mask = train_h5.create_group("miss_masks")

    val_h5 = h5py.File(cfg.dataset.val_convert_hdf5, 'w')
    val_grp = val_h5.create_group("dataset")
    val_write_count = 0
    val_grp_img = val_h5.create_group("images")
    val_grp_miss_mask = val_h5.create_group("miss_masks")

    for ds in datasets:
        coco = COCO(ds['annotation_dir'])
        ids = list(coco.imgs.keys())

        for img_index, img_id in enumerate(ids):
            ann_ids = coco.getAnnIds(imgIds=img_id)
            img_anns = coco.loadAnns(ann_ids)
            img_rec = coco.imgs[img_id]

            img = None
            mask_miss = None
            cached_img_id = None

            for data in process_image(img_id, img_index, img_rec, img_anns, ds['type']):

                if cached_img_id != data['image_id']:
                    assert img_id == data['image_id']
                    cached_img_id = data['image_id']
                    img, mask_miss, mask_all = get_masks(ds['image_dir'], cached_img_id, img_anns, coco)

                if data['is_validation']:
                    write_img(val_grp, val_grp_img, data, img, mask_miss, val_write_count, cached_img_id, val_grp_miss_mask)
                    val_write_count += 1
                else:
                    write_img(train_group, train_grp_img, data, img, mask_miss, train_write_count, cached_img_id, train_grp_miss_mask)
                    train_write_count += 1

    train_h5.close()
    val_h5.close()


if __name__ == '__main__':
    process()
