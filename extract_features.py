import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as stat
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pickle as pkl
import cv2
from os import listdir
from os.path import isfile, join


def create_label(folder):
    foldername = r'C:\Users\frede\Desktop\the_true_dataset\Scenario2\Train\%s' % folder
    filelist = os.listdir(foldername)
    label = []
    for file in filelist:
        if folder == 'Finger_failure':
            label.append(1)
        elif folder == 'CrackC':
            label.append(0)
        elif folder == 'No_failure':
            label.append(2)
    return label

def extract_images(folder):
    foldername = r'C:\Users\frede\Desktop\the_true_dataset\Scenario2\Train\%s' % folder
    onlyfiles = [f for f in listdir(foldername) if isfile(join(foldername, f))]
    # Python way
    img_list = [cv2.resize(cv2.imread(join(foldername, file), cv2.IMREAD_GRAYSCALE), (100, 100),
                           interpolation=cv2.INTER_NEAREST) for file in onlyfiles]
    img_list_flat = [img.flatten('F') for img in img_list]
    return img_list_flat

def build_failures(failures):
    enc = OneHotEncoder()
    failures_labels = failures['labels'].values
    failures_labels = np.reshape(failures_labels, (-1, 1))
    failures_labels = enc.fit_transform(failures_labels).toarray().tolist()
    result = failures.drop(['labels'], axis=1)
    result['labels'] = failures_labels

    return result

Labels0 = create_label('CrackC')
Labels1 = create_label('Finger_failure')
Labels2 = create_label('No_failure')
Labels = Labels0 + Labels1 + Labels2
images0 = extract_images('CrackC')
images1 = extract_images('Finger_failure')
images2 = extract_images('No_failure')
concatImages = images0 + images1 + images2

unique=[np.unique(img) for img in concatImages]
ELhistList = [np.bincount(img) for img in concatImages]
ELhistList = [np.concatenate((hist,np.array([0]*(256-len(hist))))) for hist in ELhistList]

stat_feature = {'mu':[],'ICA':[],'kur':[],'skew':[], 'sp':[], 'md':[], 'sd':[], 'var':[],
                '25p':[], '75p':[],'fw':[],'kstat':[],'entropy':[]}

for ELhist in ELhistList:
    PEL=ELhist/np.sum(ELhist)
    stat_feature['mu'].append(np.mean(PEL))
    stat_feature['md'].append(np.median(PEL))
    threshold=min(30,len(PEL)-1)
    stat_feature['ICA'].append(100*np.sum([PEL[i] for i in range(threshold+1)]))
    stat_feature['sd'].append(np.std(PEL))
    stat_feature['kur'].append(stat.kurtosis(PEL))
    stat_feature['skew'].append(stat.skew(PEL))
    stat_feature['var'].append(stat.variation(PEL))
    stat_feature['sp'].append(np.ptp(PEL))
    stat_feature['fw'].append(((5/100)*(np.max(PEL)))-((5/100)*(np.min(PEL))))
    stat_feature['25p'].append(np.quantile(PEL, 0.25, interpolation='lower'))
    stat_feature['75p'].append(np.quantile(PEL, 0.75, interpolation='higher'))
    stat_feature['entropy'].append(stat.entropy(PEL))
    stat_feature['kstat'].append(stat.kstat(PEL))

from sklearn.preprocessing import MinMaxScaler
dfhist = pd.DataFrame(stat_feature)
scaler = MinMaxScaler()
normdfhist= pd.DataFrame(scaler.fit_transform(dfhist))
X=pd.DataFrame(ELhistList)
Xfinal=pd.merge(X,dfhist,left_index=True,right_index=True)
sortdf = pd.DataFrame(Xfinal)
sortdf['labels']=Labels
#sortdf = build_failures(sortdf)
sortdf.to_pickle('data/training_2.pkl')
print('FILE CREATED')


