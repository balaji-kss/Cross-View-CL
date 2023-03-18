import numpy as np
import os
import json

def alignDataList(fileRoot, folder, imagesList,dataset):
    'this funciton is to make sure image list and skeleton list are aligned'
    allFiles = os.listdir(os.path.join(fileRoot, folder)) # get json files
    allFiles.sort()
    newJson_list = []
    newImage_list = []

    for i in range(0, len(imagesList)):
        if dataset == 'N-UCLA':
            json_file = imagesList[i].split('.jpg')[0] + '_keypoints.json'
        else:
            image_num = imagesList[i].split('.jpg')[0].split('_')[1]
            json_file = folder + '_rgb_0000000' + str(image_num)+'_keypoints.json'
        if json_file in allFiles:
            newJson_list.append(json_file)
            newImage_list.append(imagesList[i])

    return newJson_list, newImage_list


def getJsonData(fileRoot, folder, jsonList):
    skeleton = []
    # allFiles = os.listdir(os.path.join(fileRoot, folder))
    # allFiles.sort()
    usedID = []
    confidence = []
    mid_point_id1 = [2,3,5,6,8,9,10,12,13]
    mid_point_id2 = [3,4,6,7,1,10,11,13,14]
    for i in range(0, len(jsonList)):
        # json_file = imagesList[i].split('.jpg')[0] + '_keypoints.json'
        with open(os.path.join(fileRoot, folder, jsonList[i])) as f:
            data = json.load(f)
        # print(len(data['people']))
        if len(data['people']) != 0:
            # print('check')
            usedID.append(i)
            temp = np.asarray(data['people'][0]['pose_keypoints_2d']).reshape(25,3)
            pose = np.expand_dims(temp[:,0:2], 0)
            # midPoint = (pose[:,mid_point_id1]+pose[:,mid_point_id2])/2
            # pose = np.concatenate((pose,midPoint),1)
            s = np.array([temp[:,-1], temp[:,-1]])
            score = np.expand_dims(s.transpose(1,0), 0)
            skeleton.append(pose)
            confidence.append(score)

        else:
            continue

    skeleton = np.concatenate((skeleton))
    confidence = np.concatenate((confidence))
    return skeleton, usedID, confidence