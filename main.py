import numpy as np
import sklearn
import os

if __name__ == '__main__':
    dir_name= './SEED-IV'
    print(os.listdir(dir_name))
    sessions=os.listdir(dir_name)
    train_data=[]
    train_label=[]
    test_data=[]
    test_label=[]
    for session in sessions:
        session_dir= dir_name + '/' + session
        peoples=os.listdir(session_dir)
        for people in peoples:
            people_dir=session_dir+'/'+people+'/'
            print(people_dir)
            train_data.append(np.load(people_dir+'train_data.npy'))
            train_label.append(np.load(people_dir + 'train_label.npy'))
            test_data.append(np.load(people_dir + 'test_data.npy'))
            test_label.append(np.load(people_dir + 'test_label.npy'))

