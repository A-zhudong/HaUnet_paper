from audioop import reverse
from cProfile import label
from contextlib import nullcontext
from curses import A_CHARTEXT
from turtle import st
from xmlrpc.client import boolean
import matplotlib.pyplot as plt
import numpy as np
import random
import itertools
import cv2
import os
from functools import reduce

# transform the default O3 to O1 according to the trans_atoms and O1_atoms
def transform_o3_o1(atoms_coor, x_axis_dis, O1_atoms, x_shift):
    len_coor = len(atoms_coor)
    # print(len_coor, trans_atoms, O1_atoms)
    start = random.randint(1,len_coor-trans_atoms-O1_atoms)
    move_dis = x_axis_dis - abs(x_shift*2)
    # shift = np.concatenate((np.arange(0,move_dis+1e-4,(move_dis)/trans_atoms), np.ones(O1_atoms)*(move_dis), \
    #     np.arange(0,move_dis+1e-4,(move_dis)/trans_atoms)))
    shift = np.concatenate((np.ones(trans_atoms)*(move_dis)/trans_atoms, np.ones(O1_atoms)*(move_dis), \
        np.ones(trans_atoms)*(move_dis)/trans_atoms))
    #delete middle class, change to O3
    atoms_class = np.concatenate((np.ones(trans_atoms)*0, np.ones(O1_atoms), \
        np.ones(trans_atoms)*0))
    if len(shift)+start > len_coor:       
        shift = np.concatenate((np.zeros(start), shift[:len_coor-start]))
        atoms_class = np.concatenate((np.zeros(start), atoms_class[:len_coor-start]))
    else: 
        atoms_class = np.concatenate((np.zeros(start), atoms_class, np.zeros(len_coor-len(shift)-start)))
        shift = np.concatenate((np.zeros(start), shift, np.zeros(len_coor-len(shift)-start)))        
    after = shift + atoms_coor[:, 0].copy()
    atoms_coor[:, 0] = after.copy()
    return atoms_coor, atoms_class

# add x axis and y axis random movement to simulate real crystal
def add_shake(atoms_coor, height, waveLength):
    # print(height, waveLength)
    x_shake = np.random.normal(loc=0, scale=0.1, size=len(atoms_coor))
    if len(atoms_coor)>1:
        x = atoms_coor[:,0]
        y_sinShake = height * y_axis_dis * np.sin( 1/waveLength * x)
    else: 
        y_sinShake = 0
    y_shake = np.random.normal(loc=0, scale=0.2, size=len(atoms_coor)) + y_sinShake
    shake = np.array([x_shake,y_shake]).T
    #print(atoms_coor, shake)
    #print(shake)
    if len(atoms_coor) > 0:
        atoms_coor_shake = atoms_coor + shake
    else: atoms_coor_shake = atoms_coor
    return atoms_coor_shake, atoms_coor

# add one atom line to the simulated crystal from the upper left
def add_atom_line(x_start, y_start, x_axis_dis, possibility_atom_exist, \
    possibility_atom_exist_ori, O1_atoms, x_shift, height, waveLength, O3_O1_rate, last_exit=True, \
        before2_transform=False, before4_transform=False, atoms_class_alline=None):

    # print('in',height, waveLength)
    transform = False
    atoms_coor= []; atoms_class = []; line_atom_exist = []
    if x_start < 0:
        x_start = x_start + (abs(x_start)//x_axis_dis+1)*x_axis_dis
    if x_start > 3:
        x_start = x_start - (abs(x_start)//x_axis_dis)*x_axis_dis
    if x_start ==0: x_start = 1e-4
    x_coor = x_start; y_coor = y_start
    while x_coor < a and x_coor > 0 and y_coor < b and y_coor > 0:
        if possibility_atom_exist > random.random():
            line_atom_exist.append(True)
            atoms_coor.append([x_coor,y_coor])
        else: line_atom_exist.append(False)
        x_coor = x_coor + x_axis_dis
    atoms_coor = np.array(atoms_coor)
    # atoms_coor = np.clip(atoms_coor, 0, a)
    p = random.random()
    if y_start>0:
        if possibility_atom_exist < possibility_atom_exist_ori: # rocksalt : 2
            atoms_class = np.ones(len(atoms_coor))*2
        elif (not last_exit) and O3_O1_rate > p: # O3 O1 nomid : 0, 1, 0
            if not before2_transform: 
                atoms_coor, atoms_class = transform_o3_o1(atoms_coor, x_axis_dis, O1_atoms, x_shift)
                transform = True
            else: 
                atoms_class = np.ones(len(atoms_coor))*0
        elif (not last_exit) and O3_O1_rate < p: # O_r or O3 : 3 or 0, default to O_r
            atoms_class = np.ones(len(atoms_coor))*0
        else: # O_r or O3 : 3 or 0, default to O_r
            atoms_class = np.ones(len(atoms_coor))*2  
    atoms_coor_shake, atoms_coor = add_shake(atoms_coor, height, waveLength)
    atoms_coor_shake = np.where(atoms_coor_shake>a, a-0.1, atoms_coor_shake)
    atoms_coor_shake = np.where(atoms_coor_shake<0, 0.1, atoms_coor_shake)
    # print(np.sum(atoms_coor_shake[:,1]>b))
    return atoms_coor_shake, atoms_coor, atoms_class, transform, line_atom_exist

# define the right rocksalt and O3 in the transformed line
def trans_O3_rs(atoms_class_alline, exist_boolean, transform_boolean):
    atoms_class_alline_1 = []
    for i, line in enumerate(atoms_class_alline):
        if i<len(atoms_class_alline)-1 and transform_boolean[i+4] and exist_boolean[i+1]:
            atoms_class_alline_1.append(np.where(line==0, 2, line))
        else: atoms_class_alline_1.append(line)
    return atoms_class_alline_1

# define the right rocksalt according to the same enviroment
def get_right_rocksalt(atoms_class_alline, exist_boolean, transform_boolean):
    atoms_class_alline_1 = []
    #exist_boolean.insert(0, False)
    for i, line in enumerate(atoms_class_alline):
        #print(type(line))
        if i<len(atoms_class_alline)-1 and exist_boolean[i+1] and not transform_boolean[i+4] :
            atoms_class_alline_1.append(np.ones(line.shape)*2)
        # add edge correction from rs to O3
        elif i==0 and np.sum(atoms_class_alline[i]!=2)==0 and not exist_boolean[i+1]:
            atoms_class_alline_1.append(np.ones(line.shape)*0)
        elif i==len(atoms_class_alline)-1 and np.sum(atoms_class_alline[i]!=2)==0 and not exist_boolean[i-1]:
            atoms_class_alline_1.append(np.ones(line.shape)*0)
        else:
            atoms_class_alline_1.append(line)
    #return list(itertools.chain(*atoms_class_alline_1))
    return atoms_class_alline_1

# define the O3 atoms between two transformed lines to O1
def O1_O3_O1_correct(atoms_class_alline, exist_boolean, transform_boolean):
    for i in range(len(atoms_class_alline)):
        if i<len(atoms_class_alline)-2 and i>2 and (not exist_boolean[i-1]) and exist_boolean[i] and \
            transform_boolean[i+4-2] and transform_boolean[i+4+2] and (not transform_boolean[i+4]):
            start = max(min(np.where(atoms_class_alline[i-2]==1)[0]), min(np.where(atoms_class_alline[i+2]==1)[0]))
            stop = min(max(np.where(atoms_class_alline[i-2]==1)[0]), max(np.where(atoms_class_alline[i+2]==1)[0]))
            # print(i, start, stop)
            # print(atoms_class_alline[i])
            atoms_class_alline[i][start:stop+1] = 1
    return atoms_class_alline

# define the atoms near the scattered rocksalt to rocksalt
def scattered_rocksalt(atoms_class_alline, exist_boolean, lines_atom_exists, x_shift):
    if x_shift<0: move = 1
    else: move = -1
    for i in range(len(atoms_class_alline)):
        #consider two edge situations and middle situations
        if i == 0 and not exist_boolean[i]:
            for idx in [index for index,x in enumerate(lines_atom_exists[i]) if x==True]:
                if idx == 0: atoms_class_alline[i+1][idx] = 2
                elif idx<len(atoms_class_alline[i+1]):
                    if atoms_class_alline[i+1][idx]!=1: atoms_class_alline[i+1][idx] = 2
                    if atoms_class_alline[i+1][idx-move]!=1:  atoms_class_alline[i+1][idx-move] = 2
        elif i == len(atoms_class_alline)-1 and not exist_boolean[i]:
            for idx in [index for index,x in enumerate(lines_atom_exists[i]) if x==True]:
                if idx == len(atoms_class_alline[i-1])-1: atoms_class_alline[i-1][idx] = 2
                elif idx<len(atoms_class_alline[i-1])-1:
                    if atoms_class_alline[i-1][idx]!=1: atoms_class_alline[i-1][idx] = 2
                    if atoms_class_alline[i-1][idx+move]!=1: atoms_class_alline[i-1][idx+move] = 2       
        elif not exist_boolean[i]:
            for idx in [index for index,x in enumerate(lines_atom_exists[i]) if x==True]:
                if idx == 0: atoms_class_alline[i+1][idx] = 2
                elif idx<len(atoms_class_alline[i+1])-1:
                    if atoms_class_alline[i+1][idx]!=1: atoms_class_alline[i+1][idx] = 2
                    if atoms_class_alline[i+1][idx-move]!=1: atoms_class_alline[i+1][idx-move] = 2

                if idx == len(atoms_class_alline[i-1])-1: atoms_class_alline[i-1][idx] = 2
                elif idx<len(atoms_class_alline[i-1])-1:
                    if atoms_class_alline[i-1][idx]!=1: atoms_class_alline[i-1][idx] = 2
                    if atoms_class_alline[i-1][idx+move]!=1: atoms_class_alline[i-1][idx+move] = 2     
    return atoms_class_alline

# delete atoms from all atoms randomly
def del_list_from_list(src, idxs):
    idxs.sort(reverse=True)
    for idx in idxs:
        src.pop(idx)
    return src

# generate edge dislocation can use or
# 1)delete part of one line and change the location of atoms near the edge
# 2)delete and change the class of atoms accordingly
def generate_edge_dislocation(atoms_coor_shake_alline, atoms_class_alline, exist_boolean, x_axis_dis, y_axis_dis, a,\
    possibility_disloaction_exist=1, ifc_rge=8, inside_O=True, num_dis = 0):
    # print(len(atoms_coor_shake_alline))
    if possibility_disloaction_exist >= random.random():
        if inside_O: edge_scale = 1
        else: edge_scale = 2
        possi_i = []
        for i in range(len(exist_boolean)):
            position_condition = i>1 and i<len(exist_boolean)-4
            start = i-int((ifc_rge+2)/edge_scale) if i-int((ifc_rge+2)/edge_scale)>=0 else 0
            stop = i+int(ifc_rge/edge_scale) if i+int(ifc_rge/edge_scale)<=len(exist_boolean)-1 else len(exist_boolean)-1
            # print('i value:', i)
            # print('start, stop: {} {}'.format(start, stop))
            if start<=i-2/edge_scale and stop>=i+4/edge_scale and exist_boolean[i]:
                if inside_O:
                    con_ahead = (i-start)%2 ; con_back = (stop-i)%2
                    y_is = [exist_boolean[idx] for idx in range(start-con_ahead, stop-con_back+1, 2)]
                    y_not = [not exist_boolean[idx] for idx in range(start-con_ahead+1, stop-con_back, 2)]
                    exist_condition = reduce(lambda x,y:x*y, y_is) and reduce(lambda x,y:x*y, y_not)
                else:
                    y_is = [exist_boolean[idx] for idx in range(start, stop+1)]
                    exist_condition = reduce(lambda x,y:x*y, y_is)
                # print('inside start, stop: {} {}'.format(start, stop))
                # print(y_is, y_not)
            else: exist_condition = False
            edge_exist_condition = position_condition and exist_condition
            if edge_exist_condition: possi_i.append(i)
        if len(possi_i) > 0:
            y_index = random.sample(possi_i, 1)[0]; x_length = len(atoms_coor_shake_alline[y_index])
            x_index = random.sample(range(int(x_length/5), int(x_length*4/5)),1)[0]
            # print(y_index, x_index)
            # print(atoms_coor_shake_alline[y_index][x_index])
            x_trun = atoms_coor_shake_alline[y_index][x_index][0]
            start = y_index+int((-ifc_rge+2)/edge_scale) if y_index+int((-ifc_rge+2)/edge_scale)>=0 else 0
            stop = y_index+int(ifc_rge/edge_scale) if y_index+int(ifc_rge/edge_scale)<=len(exist_boolean)-1 else len(exist_boolean)-1
            #removee center atoms
            if inside_O:                
                atoms_coor_shake_alline[y_index+2] =np.delete(atoms_coor_shake_alline[y_index+2],[x_index-1,x_index,x_index+1,x_index+2],axis=0)
                print(atoms_coor_shake_alline[y_index+2].shape)
                atoms_class_alline[y_index+2] = np.delete(atoms_class_alline[y_index+2],[x_index-1,x_index,x_index+1,x_index+2])
            else:
                atoms_coor_shake_alline[y_index+1] =np.delete(atoms_coor_shake_alline[y_index+1],[x_index-1,x_index,x_index+1,x_index+2],axis=0)
                print(atoms_coor_shake_alline[y_index+1].shape)
                atoms_class_alline[y_index+1] = np.delete(atoms_class_alline[y_index+1],[x_index-1,x_index,x_index+1,x_index+2])
            for idx in range(start,stop+1):
                if inside_O:
                    if idx == y_index or idx == y_index+1:
                        #removee center atoms
                        atoms_coor_shake_alline[idx] = atoms_coor_shake_alline[idx][x_index+2:]
                        atoms_class_alline[idx] = atoms_class_alline[idx][x_index+2:]
                        if idx == y_index: atoms_coor_shake_alline[y_index][0][1] = atoms_coor_shake_alline[y_index][0][1] - y_axis_dis*2/5/edge_scale
                        # make the up near line bend
                    else:
                        # here to start
                        if y_index-idx+1 != 0: y_shift_idx = y_axis_dis/(y_index-idx+1)
                        else: y_shift_idx = 0
                        for coor in atoms_coor_shake_alline[idx]:
                            if coor[0]<x_trun: coor[1] = coor[1]-y_shift_idx
                        # print(idx, y_shift_idx)
                    
                else:
                    if idx == y_index:
                        atoms_coor_shake_alline[idx] = atoms_coor_shake_alline[idx][x_index+2:]
                        atoms_class_alline[idx] = atoms_class_alline[idx][x_index+2:]
                        atoms_coor_shake_alline[y_index][0][1] = atoms_coor_shake_alline[y_index][0][1] - y_axis_dis*2/5/edge_scale
                    else:
                        y_shift_idx = y_axis_dis/(y_index+1/edge_scale-idx)/edge_scale**2
                        for coor in atoms_coor_shake_alline[idx]:
                            if coor[0]<x_trun: coor[1] = coor[1]-y_shift_idx
                        # print(idx, y_shift_idx)
            # make the down near line bend
            for value in atoms_coor_shake_alline[y_index+int(2/edge_scale)]:
                if value[0]-x_trun > 0:
                    value[1] = value[1]+y_axis_dis*2/5/edge_scale
                    break
            num_change_class = 5; 
            class_start = x_trun-num_change_class*x_axis_dis if x_trun-num_change_class*x_axis_dis>0 else 0
            class_stop = x_trun+num_change_class*x_axis_dis if x_trun-num_change_class*x_axis_dis<a else a
            for index in range(y_index-int(0/edge_scale), y_index+int(2/edge_scale)+1):
            #change to center
            # for index in range(y_index, y_index+int(2/edge_scale)+1):
                for coor_idx, line_coor in enumerate(atoms_coor_shake_alline[index]):
                    if line_coor[0]>=class_start and line_coor[0]<=class_stop:
                        atoms_class_alline[index][coor_idx] = 3
            num_dis = num_dis + 1
            print('num_dis: ', num_dis)
    return num_dis
    # return atoms_coor_shake_alline

#generate masks for all atoms images
def check_around(coor, mask, img, label_i, coor_masked, left=False, right=False, up=False, down=False):
    # print(np.sum(coor_masked))
    if coor[0]-1 > -1 :
        if img[coor[0]-1, coor[1]]>0 and (not coor_masked[coor[0]-1, coor[1]]) and (not right):
            # left = True; right = False
            coor_masked[coor[0]-1, coor[1]] = 1
            mask[coor[0]-1, coor[1]] = label_i
            check_around([coor[0]-1, coor[1]], mask, img, mask[coor[0]-1, coor[1]], coor_masked)#, left=left, right=right)
    if coor[0]+1 < mask.shape[0]:
        if img[coor[0]+1, coor[1]]>0 and (not coor_masked[coor[0]+1, coor[1]]) and (not left):
            # right = True; left = False
            coor_masked[coor[0]+1, coor[1]] = 1
            mask[coor[0]+1, coor[1]] = label_i
            check_around([coor[0]+1, coor[1]], mask, img, mask[coor[0]+1, coor[1]], coor_masked)#, left=left, right=right)
    if coor[1]-1 > -1:
        if img[coor[0], coor[1]-1]>0 and (not coor_masked[coor[0], coor[1]-1]) and (not up):
            # down = True; up = False
            coor_masked[coor[0], coor[1]-1] =1
            mask[coor[0], coor[1]-1] = label_i
            check_around([coor[0], coor[1]-1], mask, img, mask[coor[0], coor[1]-1], coor_masked)#, up=up, down=down)
    if coor[1]+1 < mask.shape[1]:
        if img[coor[0], coor[1]+1]>0 and (not coor_masked[coor[0], coor[1]+1]) and (not down):
            # up = True; down = False
            coor_masked[coor[0], coor[1]+1] = 1
            mask[coor[0], coor[1]+1] = label_i
            check_around([coor[0], coor[1]+1], mask, img, mask[coor[0], coor[1]+1], coor_masked)#, up=up, down=down)

# generate mask according to nearest neighbors
def generate_maskImage(img, label, num, storePath):
    # import sys
    # sys.setrecursionlimit(1500)
    shape = img.shape
    mask = np.zeros(shape)
    coor_masked = np.zeros(shape)
    coors = label[:,:2]
    labels = label[:,2]
    # print(label.shape)
    # sum_ori = np.sum(coor_masked)
    for i, coor in enumerate(coors):
        # print(img.shape, coor, labels[i])
        if img[coor[0], coor[1]] == 0 : print('exist not good')
        # print(img[coor[0], coor[1]], labels[i])
        mask[coor[0],coor[1]] = labels[i] + 1
        coor_masked[coor[0],coor[1]] = 1
        check_around(coor, mask, img, labels[i]+1, coor_masked)
    # sum_mask  = np.sum(coor_masked)
    # while sum_ori < sum_mask:
    #     sum_ori = sum_mask
    #     for x_idx in range(len(coor_masked)):
    #         for y_idx in range(len(coor_masked[0])):
    #             if coor_masked[x_idx, y_idx] == 1:
    #                 check_around(mask, img, coor_masked, [x_idx,y_idx])
    #     sum_mask  = np.sum(coor_masked)
    np.save(os.path.join(storePath, f'{num}_mask.npy'), mask)
    # print(np.sum(mask>0)/shape[0]/shape[1])

#generate image and label
def generate_image_label(a, b, x, y, coor_scale, storePath, mask_storePath, num, atoms_class_alline):
    plt.figure(figsize=(15, 10))
    plt.xlim(0, a)
    plt.ylim(0, b)
    # plt.yticks(rotation=90)
    plt.axis('off')
    plt.scatter(x,y, s=5/(coor_scale**2))#, c=atoms_class_alline)
    plt.savefig(os.path.join(storePath,'{}.png'.format(num)), \
        bbox_inches='tight', pad_inches=0, dpi=int(1*60))
    #plt.show()
    plt.close()
    # print(x[-10:]*697/450, y[:10])

    img = cv2.imread(os.path.join(storePath,'{}.png'.format(num)), cv2.IMREAD_GRAYSCALE)
    img = abs(255-img)
    # print(img[2,2])
    cv2.imwrite(os.path.join(storePath,'{}.png'.format(num)), img)
    ishape = img.shape
    # print(ishape)
    x_1 = np.round(x*ishape[1]/a); y_1 = ishape[0]-np.round(y*ishape[0]/b)
    x_1 = np.where(x_1>=ishape[1], ishape[1]-1, x_1)
    y_1 = np.where(y_1>=ishape[0], ishape[0]-1, y_1)
    # print(x_1[-10:], y_1[-10:])
    # print(np.max(x_1), np.max(y_1))
    if num%1000==0: print(num)

    # cv2.imshow('black_white',img)
    # cv2.waitKey(2000)
    
    label = np.concatenate((y_1.reshape(1,-1), x_1.reshape(1,-1), np.array(atoms_class_alline).reshape(1,-1)))
    np.savetxt(os.path.join(storePath,'{}.txt'.format(num)), label.T)
    # generate_maskImage(img, label.T.astype(int), num, mask_storePath)
    num = num + 1
    return num

storePath = './hardAttention_behindpart_angstrom_withNoise_4w_edge_dis_noCenter_npyform'
mask_storePath = './Pytorch-UNet-master_atombulk/data/masks'
if not os.path.exists(storePath): os.mkdir(storePath)
# if not os.path.exists(mask_storePath): os.mkdir(mask_storePath)

num = 0
num_dis = 0
for coor_scale in [0.5, 1, 1.5, 2, 2.5]:
# for coor_scale in [0.5, 1, 1.5]:
# for coor_scale in [1]:
    for possibility_line_exist in[0.2, 0.35]:
    # for possibility_line_exist in[1]:
        #set random possibility
        possibility_atom_exist_ori = 1
        
        #set box size
        a = 150*coor_scale
        # b = 100*coor_scale
        b = 150*coor_scale

        #set distance between x axis and y axis
        x_axis_dis = 2.4
        y_axis_dis = 2.4

        # x_shift = 0.75
        x_start_ori, y_start_ori = 2, b-2

        #set transform and O1 atoms
        trans_atoms = int(a/x_axis_dis/20)
        # O1_atoms = np.random.randint(int(a/x_axis_dis/10),int(a/x_axis_dis/2))
        
        O3_O1_rate = 0.65
        # rate for edge dislocation
        possibility_disloaction_exist = 0.2
        # possibility_disloaction_exist = 1
        inside_O_possible = 0.5


        for i in range(500):
        # for i in range(1000):
            seed = i
            random.seed(seed)
            np.random.seed(seed)
            height = random.uniform(1/3, 1/2)
            waveLength = random.uniform(18,40)
            #set distance between x axis and y axis
            # x_axis_dis = 3
            # y_axis_dis = 3
            y_axis_dis = y_axis_dis + random.uniform(-0.05,0.05)
            for x_shift in [x_axis_dis/4, 0-x_axis_dis/4]:
                # print('out', height, waveLength)
                O1_atoms = np.random.randint(int(a/x_axis_dis/10),int(a/x_axis_dis/2))
                # print(O1_atoms)
                #record the current number of lines
                num_lines = 4
                exist_boolean = []; transform_boolean = []
                #add four lines that didn't exist 
                transform_boolean.extend([False, False, False, False])

                #when the front one existed, the back one considered possibility of line_existing
                atoms_coor_shake_alline = []; atoms_coor_alline = []; 
                atoms_class_alline = []; lines_atom_exists = []
                atoms_coor_shake, atoms_coor, atoms_class, transform, line_atom_exist = add_atom_line(x_start_ori, y_start_ori, x_axis_dis, possibility_atom_exist_ori,\
                    possibility_atom_exist_ori, O1_atoms, x_shift, height, waveLength, O3_O1_rate, True, False, False)
                num_lines = num_lines + 1
                atoms_class_alline.append(atoms_class)
                transform_boolean.append(transform)
                lines_atom_exists.append(line_atom_exist)
                # print(atoms_coor.shape)
                atoms_coor_shake_alline.append(atoms_coor_shake); atoms_coor_alline.extend(atoms_coor)
                #print(atoms_coor_shake_all)

                x_start = x_start_ori - x_shift; y_start = y_start_ori - y_axis_dis
                last_exit = True
                exist_boolean.append(last_exit)
                while y_start > 2:
                    # print('out', height, waveLength)
                    if possibility_line_exist < random.uniform(0,1) and last_exit:
                        possibility_atom_exist = random.choice([0.05, 0.15, 0.205])
                        atoms_coor_shake, atoms_coor, atoms_class, transform, line_atom_exist = add_atom_line(x_start, y_start, x_axis_dis, \
                            possibility_atom_exist, possibility_atom_exist_ori, O1_atoms, x_shift, height, waveLength, O3_O1_rate, last_exit, \
                                transform_boolean[num_lines-2], transform_boolean[num_lines-4], atoms_class_alline)
                        last_exit = False
                    else: 
                        possibility_atom_exist = possibility_atom_exist_ori
                        atoms_coor_shake, atoms_coor, atoms_class, transform, line_atom_exist = add_atom_line(x_start, y_start, x_axis_dis, \
                            possibility_atom_exist, possibility_atom_exist_ori, O1_atoms, x_shift, height, waveLength,  O3_O1_rate, last_exit, \
                                transform_boolean[num_lines-2], transform_boolean[num_lines-4], atoms_class_alline)
                        last_exit = True
                    exist_boolean.append(last_exit)    
                    num_lines = num_lines + 1
                    transform_boolean.append(transform)
                    atoms_class_alline.append(atoms_class)
                    lines_atom_exists.append(line_atom_exist)
                    atoms_coor_shake_alline.append(atoms_coor_shake); atoms_coor_alline.extend(atoms_coor)
                    x_start = x_start - x_shift; y_start = y_start - y_axis_dis
                    possibility_atom_exist = possibility_atom_exist_ori
                    # break
                # print(len(atoms_class_alline))
                #atoms_class_alline = np.array(sum(atoms_class_alline,[])).astype(np.int8)

                
                # atoms_coor_shake_alline = np.array(atoms_coor_shake_alline)
                #print(np.array(atoms_class_alline).astype(np.int8))
                # print(atoms_coor_shake_alline.shape)

                # print(transform_o3_o1([1,1,1,1,1,1], 3))
                atoms_class_alline = O1_O3_O1_correct(atoms_class_alline, exist_boolean, transform_boolean)
                atoms_class_alline = get_right_rocksalt(atoms_class_alline, exist_boolean, transform_boolean)
                atoms_class_alline = trans_O3_rs(atoms_class_alline, exist_boolean, transform_boolean)
                atoms_class_alline = scattered_rocksalt(atoms_class_alline, exist_boolean, lines_atom_exists, x_shift)
                # generate edge dislocation
                if inside_O_possible > random.random(): inside_O = True
                else: inside_O = False
                num_dis = generate_edge_dislocation(atoms_coor_shake_alline, atoms_class_alline, exist_boolean, x_axis_dis, y_axis_dis, a,\
                    possibility_disloaction_exist=possibility_disloaction_exist, inside_O=inside_O, num_dis=num_dis)

                atoms_class_alline_1 = []
                for line in atoms_class_alline:
                    atoms_class_alline_1.extend(line)
                atoms_class_alline = atoms_class_alline_1
                atoms_coor_shake_alline = np.array([x for xx in atoms_coor_shake_alline for x in xx])
                del_idxs = random.sample(range(len(atoms_class_alline)),int(len(atoms_class_alline)/50))
                atoms_class_alline = del_list_from_list(atoms_class_alline, del_idxs)
                atoms_coor_shake_alline = np.array(del_list_from_list(list(atoms_coor_shake_alline), del_idxs))
                x = atoms_coor_shake_alline[:,0]; y = atoms_coor_shake_alline[:,1]
                x[0] = x_start_ori; y[0] = y_start_ori
                #save fig to hard attention image
                # image_size = 560
                # img = np.zeros((image_size,image_size), dtype=int)
                # x = np.round(x/a*image_size).astype(int)
                # x = np.where(x>0, x, 0)
                # x = np.where(x<image_size, x, image_size-1)
                # y = np.round(y/b*image_size).astype(int)
                # y = np.where(y>0, y, 0)
                # y = np.where(y<image_size, y, image_size-1)
                # # print(transform_boolean[4:])
                # # print(a,b)
                # np.savetxt(os.path.join(storePath,'{}.txt'.format(num)),np.concatenate((y.reshape(1,-1), x.reshape(1,-1), np.array(atoms_class_alline).reshape(1,-1))))
                # img[x,y] = 1
                # np.save(os.path.join(storePath,'{}.npy'.format(num)), img)
                # img = np.zeros((image_size,image_size))
                # num = num + 1
                # np.savetxt(os.path.join(storePath,'{}.txt'.format(num)),np.concatenate((y.reshape(1,-1), abs(a-x).reshape(1,-1), np.array(atoms_class_alline).reshape(1,-1))))
                # img[(-x+a).astype(int),y] = 1
                # np.save(os.path.join(storePath,'{}.npy'.format(num)), img)
                # img = np.zeros((image_size,image_size))
                # num = num + 1
                # np.savetxt(os.path.join(storePath,'{}.txt'.format(num)),np.concatenate((abs(b-y).reshape(1,-1), x.reshape(1,-1), np.array(atoms_class_alline).reshape(1,-1))))
                # img[x,(b-y).astype(int)] = 1
                # np.save(os.path.join(storePath,'{}.npy'.format(num)), img)
                # img = np.zeros((image_size,image_size))
                # num = num + 1
                # np.savetxt(os.path.join(storePath,'{}.txt'.format(num)),np.concatenate((abs(b-y).reshape(1,-1), abs(a-x).reshape(1,-1), np.array(atoms_class_alline).reshape(1,-1))))
                # img[(a-x).astype(int),(b-y).astype(int)] = 1
                # np.save(os.path.join(storePath,'{}.npy'.format(num)), img)
                # img = np.zeros((image_size,image_size))
                # num = num + 1
                num = generate_image_label(a, b, x, y, coor_scale, storePath, mask_storePath, num, atoms_class_alline)
                num = generate_image_label(a, b, abs(np.max(x)-x), y, coor_scale, storePath, mask_storePath, num, atoms_class_alline)
                num = generate_image_label(a, b, x, abs(np.max(y)-y), coor_scale, storePath, mask_storePath, num, atoms_class_alline)
                num = generate_image_label(a, b, abs(np.max(x)-x), abs(np.max(y)-y), coor_scale, storePath, mask_storePath, num, atoms_class_alline)
            # break
np.savetxt(os.path.join(storePath,'index.txt'), np.arange(num).astype(np.int32))
                
