"""
Helper functions for LiTS
"""
import numpy as np
from scipy import ndimage as ndi
from scipy import misc as mc


 #Computed in liver_data_preparations
MIN_BOUND = -1000.0 #Everything below: Water
MAX_BOUND = 400.0 #Everything above corresponds to bones
PIXEL_MEAN = 0.338

def set_bounds(image):
    MIN_BOUND = -1000.0 #Everything below: Water
    MAX_BOUND = 400.0
    image[image>MAX_BOUND]=MAX_BOUND
    image[image<MIN_BOUND]=MIN_BOUND
    return image

def normalize(image,use_bd=False,zero_center=False):
    if not use_bd:
        MIN_BOUND = np.min(image)
        MAX_BOUND = np.max(image)
    else:
        MIN_BOUND = -1000.0 #Everything below: Water
        MAX_BOUND = 400.0
        image = set_bounds(image)
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    if zero_center:
        image = image - PIXEL_MEAN
    return image

#Convert to 5dim-shape
def load_and_prepare(to_prep,wlist=None,prep=False,downs=0):
    data_stack = []
    if wlist == None:
        for nift in to_prep:
            to_app=np.expand_dims(np.transpose(normalize(np.load(nift),zero_center=True),(2,0,1)),axis=0)
            data_stack.append(np.expand_dims(to_app,axis=0))
    else:
        for choice in wlist:
            if downs:
                downsampled_im = ndi.interpolation.zoom(np.load(to_prep[choice]),downs)
            else:
                downsampled_im = np.load(to_prep[choice])
            if prep:
                to_app=np.expand_dims(np.transpose(normalize(downsampled_im,zero_center=True),(2,0,1)),axis=0)
            else:
                to_app=np.expand_dims(np.transpose(downsampled_im,(2,0,1)),axis=0)
            data_stack.append(np.expand_dims(to_app,axis=0))
    data_stack = np.vstack(data_stack)
    return data_stack
    del data_stack,to_app

def repurpose_files(INPUT_FOLDER,INPUT_FOLDER2,data,indices,volumes):
    for k in xrange(len(data)):
        os.rename(os.path.join(INPUT_FOLDER,data[indices[k]]),os.path.join(INPUT_FOLDER2,"Pure_Skeleton-"+volumes[k].split(".")[0]+".png"))

def resize_all():
    for i in xrange(len(volumes)):

        kk = lhf.load_and_prepare(volumes,[i],downs=0.25)
        kk.shape
        kk2 = lhf.load_and_prepare(segmentations,[i],downs=0.25)
        np.save(INPUT_FOLDER2+"/"+volumes_temp[i].split(".")[0],kk)
        np.save(INPUT_FOLDER2+"/"+segmentations_temp[i].split(".")[0],kk2)
        print "Finished resizing: {}/{}".format(i,len(volumes))
        print "With volume shape: ",kk.shape
        print "With segmentation shape: ",kk2.shape
        del kk,kk2
        gc.collect()
        p = os.system('echo %s|sudo -S %s' % (sudoPassword, command))

def downsample():
    for i in xrange(len(volumes)):
        v = np.load(volumes[i])
        s = np.load(segmentations[i])
        m = ndi.interpolation.zoom(v[0,0,:,:,:],0.25)
        n = ndi.interpolation.zoom(s[0,0,:,:,:],0.25)
        im_vol = np.expand_dims(np.expand_dims(m,axis=0),axis=0)
        im_seg = np.expand_dims(np.expand_dims(n,axis=0),axis=0)
        np.save(INPUT_FOLDER2+"/"+volumes_temp[i].split(".")[0],im_vol)
        np.save(INPUT_FOLDER2+"/"+segmentations_temp[i].split(".")[0],im_seg)
        del v,s,im_vol,im_seg,m,n
        p = os.system('echo %s|sudo -S %s' % (sudoPassword, command))
        print "Finished: ",i

""""""""""""""""""""""""""""""""""""""""""""""""
""" Training functions """
""""""""""""""""""""""""""""""""""""""""""""""""
def patch_slice(data,patch_dim=4,f_size=3,z_slice=True,dorand=False,n_patches=10,seeed=111):
    """
    -data:      Input data to be patched. Needs to be of dimension (bs,ch,depth(z),width(x),height(y))
    -patch_dim: Number of patches you want to divide the original data into.
    -f_size:    Filter size of network to ensure meaningful convolution
    -z_slice:   Only slice in z-direction or blockwise. Changes the purpose of patch_dim to
                either actual patch size or axis-divider!
    -dorand:    Randomly get patches.
    -n_patches: # of patches to be randomly selected

    Note: For z_slice=True: as long as the dimension of the patch is bigger than f_size, no padding will be done!
    """
    np.random.seed(seeed)
    dshape=data.shape
    n2 = dshape[2]/patch_dim
    n3 = dshape[3]/patch_dim
    n4 = dshape[4]/patch_dim
    patch_list=[]
    if not dorand:
        if not z_slice:
            for i in xrange(0, dshape[2]+n2, n2):
                for j in xrange(0, dshape[3]+n3, n3):
                    for k in xrange(0, dshape[4]+n4, n4):
                        get_patch = data[:,:,i:i+n2,j:j+n3,k:k+n4]
                        if 0 not in get_patch.shape:
                            if all(l>f_size for l in get_patch.shape[2:]):
                                patch_list.append(get_patch)
                            else:
                                for m in xrange(3):
                                    p = [((0,0),(0,0),(0,dshape[2]/patch_dim-get_patch.shape[2]),(0,0),(0,0)),
                                         ((0,0),(0,0),(0,0),(0,dshape[3]/patch_dim-get_patch.shape[3]),(0,0)),
                                         ((0,0),(0,0),(0,0),(0,0),(0,dshape[4]/patch_dim-get_patch.shape[4])),]
                                    if get_patch.shape[m+2]<=f_size:
                                        get_patch=np.pad(get_patch,p[m],"constant",constant_values=np.max(data))
                                patch_list.append(get_patch)
        else:
            for i in xrange(0,dshape[2]+patch_dim,patch_dim):
                get_patch = data[:,:,i:i+patch_dim,:,:]
                if 0 not in get_patch.shape:
                    if get_patch.shape[2]>f_size:
                        patch_list.append(get_patch)
                    else:
                        get_patch=np.pad(get_patch,((0,0),(0,0),(0,patch_dim-get_patch.shape[2]),(0,0),(0,0)),"constant",constant_values=np.max(data))
                        patch_list.append(get_patch)
    else:
        if not z_slice:
            for i in xrange(n_patches):
                a=np.random.randint(0,dshape[2])
                b=np.random.randint(0,dshape[3])
                c=np.random.randint(0,dshape[4])
                get_patch= data[:,:,a:a+n2,b:b+n3,c:c+n4]
                if 0 not in get_patch.shape:
                    if all(l>f_size for l in get_patch.shape[2:]):
                        patch_list.append(get_patch)
                    else:
                        for m in xrange(3):
                            p = [((0,0),(0,0),(0,dshape[2]/patch_dim-get_patch.shape[2]),(0,0),(0,0)),
                                 ((0,0),(0,0),(0,0),(0,dshape[3]/patch_dim-get_patch.shape[3]),(0,0)),
                                 ((0,0),(0,0),(0,0),(0,0),(0,dshape[4]/patch_dim-get_patch.shape[4])),]
                            if get_patch.shape[m+2]<=f_size:
                                get_patch=np.pad(get_patch,p[m],"constant",constant_values=np.max(data))
                        patch_list.append(get_patch)
        else:
            for i in xrange(n_patches):
                z = np.random.randint(0,dshape[2])
                get_patch = data[:,:,z:z+patch_dim,:,:]
                if 0 not in get_patch.shape:
                    if get_patch.shape[2]>f_size and get_patch.shape[2]==patch_dim:
                        patch_list.append(get_patch)
                    else:
                        get_patch=np.pad(get_patch,((0,0),(0,0),(0,patch_dim-get_patch.shape[2]),(0,0),(0,0)),"constant",constant_values=np.max(data))
                        patch_list.append(get_patch)

    return patch_list
    del patch_list,get_patch

def augment_data(img,mode=["rot"],seeed=111):
    np.random.seed(seeed)
    t_rot=np.random.randint(-180,180,1)
    augmention=np.zeros_like(img)
    if "rot" in mode:
        for i in xrange(img.shape[2]):
            augmention[0,0,i,:,:]=mc.imrotate(img[0,0,i,:,:],t_rot)
    if "distort" in mode:
        print "Not implemented yet!"
    return augmention

def fit_dimensions(img):
    return np.expand_dims(np.expand_dims(np.transpose(img,(2,0,1)),axis=0),axis=0)

def quick_structure_check(volumes,segmentations):
    col=[]
    for i in xrange(len(segmentations)):
        kk = np.load(segmentations[i]).shape
        ll = np.load(volumes[i]).shape
        col.append(kk)
    f=np.vstack(col)
    return f
