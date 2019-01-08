"""
@author: Karsten Roth
@title: Prepare CT-Scan data/plots
"""
import cPickle
from PIL import Image
from resizeimage import resizeimage
import numpy as np
import pandas as pd
import os
import time
import theano
from theano import tensor as T
import csv
from scipy import misc as mc
from matplotlib import pyplot as plt
import lasagne
from lasagne.layers import concat,InputLayer, ConcatLayer, Pool2DLayer, Deconv2DLayer, Conv2DLayer, DenseLayer
from lasagne.layers import ReshapeLayer, DimshuffleLayer, NonlinearityLayer, DropoutLayer
import network_library_functions as network
import nibabel as nib
import scipy.ndimage
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import gc
import own_layers
from scipy import ndimage as ndi
import subprocess


""""""""""""""""""""""""
""" Load pictures """
""""""""""""""""""""""""
os.chdir("/home/nnk/Bachelor/LiTS")

#Need to open external drive once for it to work
INPUT_FOLDER = "/media/nnk/Z-NochMehrPlatz/LiTS_data/LiTS-Training_data"
data = os.listdir(INPUT_FOLDER)
data.sort()
split=len(data)/2
segmentations = data[0:split]
volumes = data[split:]

def get_data(list,ret_im=False):
    to_return=[]
    for path in list:
        if not ret_im:
            to_return.append(nib.load(INPUT_FOLDER+"/"+path))
        else:
            to_return.append(nib.load(INPUT_FOLDER+"/"+path).get_data())
    return to_return

def plot_all_layers(img):
    for i in xrange(img.shape[2]):
        plt.imshow(img[:,:,i],cmap="gray_r")
        plt.title("Image: "+str(i))
        plt.show()

volume_list=get_data(volumes)
segmentation_list=get_data(segmentations)
#return_code = subprocess.call("sync; echo 3 | sudo tee /proc/sys/vm/drop_caches", shell=True)

check_shape=[]
for i in xrange(len(volume_list)):
    tt = np.asarray(volume_list[i].dataobj).shape
    check_shape.append(tt)
    del tt
    #volume_list[i].uncache()
gc.collect()

""""""""""""""""""""""""""""""""""""""""""""""""
""" 3D Reconstructing the picture """
""""""""""""""""""""""""""""""""""""""""""""""""
#3D plot segmentation of liver and nodules
def plot_3d_seg(image,name,threshold=1,save=False):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    check = np.max(np.unique(image))>1
    verts, faces = measure.marching_cubes(image,threshold-1)
    if check:
        verts2, faces2 = measure.marching_cubes(image,threshold)
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.3)
    if check:
        mesh2 = Poly3DCollection(verts2[faces2], alpha=0.3)
    face_color = [1, 0.2, 0.2]
    if check:
        face_color2 = [0.3, 0.3, 1]
    mesh.set_facecolor(face_color)
    if check:
        mesh2.set_facecolor(face_color2)
    ax.add_collection3d(mesh)
    if check:
        ax.add_collection3d(mesh2)
    mesh_z = np.mean(image,axis=2)
    #mesh_y = np.mean(image,axis=1)
    #mesh_x = np.mean(image,axis=0)

    X=np.linspace(0,image.shape[0]-1,image.shape[0])
    Y=np.linspace(0,image.shape[1]-1,image.shape[1])
    Z=np.linspace(0,image.shape[2]-1,image.shape[2])
    #a,b=np.meshgrid(Y,Z)
    c,d=np.meshgrid(X,Y)
    #e,f=np.meshgrid(X,Z)
    cest = ax.contourf(c,d, np.transpose(mesh_z), zdir='z', offset=0, cmap="Blues")
    #cest = ax.contourf(np.transpose(mesh_x),b,a,zdir='x', offset=0, cmap="Greys")
    #cest = ax.contourf(e,np.transpose(mesh_y),f,zdir="y", offset=image.shape[1], cmap="Greys")
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_ylim(0, image.shape[1])
    ax.set_xlim(0, image.shape[0])
    ax.set_zlim(0, image.shape[2])
    ax.set_title(name+": 3D nodules and liver")
    if save:
        fig.savefig("/home/nnk/Bachelor/LiTS/Liver_segmentations/"+name+"_3D_nodules_and_liver.png", bbox_inches='tight')
    plt.close(fig)
    del mesh, verts, faces, face_color
    if check:
        del mesh2, verts2, faces2, face_color2

#3D Plot the complete image
def plot_3d_vol(image,name="Check",threshold=320,save=False):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    verts, faces = measure.marching_cubes(image,threshold-1)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.3)
    face_color = [0,0,0]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, image.shape[0])
    ax.set_ylim(0, image.shape[1])
    ax.set_zlim(0, image.shape[2])
    ax.set_title(name+"_3D_Volume_Scan")
    if save:
        fig.savefig("/home/nnk/Bachelor/LiTS/Skeleton_volume/"+name+"_3D_Volume_Scan.png", bbox_inches='tight')
    del mesh, verts, faces
    plt.close(fig)

def plot_3d_all(image,segm,name="Complete",threshold_bones=320,save=False):

    check = np.max(np.unique(segm))>1
    print "Finding marching cubes..."
    verts, faces = measure.marching_cubes(segm,0)
    if check:
        verts2, faces2 = measure.marching_cubes(segm,1)
    verts_vol, faces_vol=measure.marching_cubes(image,threshold_bones)
    fig = plt.figure(figsize=(15, 20))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    print "Computing polygons..."
    mesh = Poly3DCollection(verts[faces], alpha=0.4)
    if check:
        mesh2 = Poly3DCollection(verts2[faces2], alpha=0.7)
    mesh_vol = Poly3DCollection(verts_vol[faces_vol],alpha=0.25)
    print "Plotting..."
    face_color = [1, 0.2, 0.2]
    if check:
        face_color2 = [0.3, 0.3, 1]
    face_color_vol = [0,0,0]
    mesh.set_facecolor(face_color)
    if check:
        mesh2.set_facecolor(face_color2)
    mesh_vol.set_facecolor(face_color_vol)
    ax.add_collection3d(mesh)
    if check:
        ax.add_collection3d(mesh2)
    ax.add_collection3d(mesh_vol)
    mesh_z = np.mean(segm,axis=2)
    #mesh_y = np.mean(image,axis=1)
    #mesh_x = np.mean(image,axis=0)

    X=np.linspace(0,image.shape[0]-1,image.shape[0])
    Y=np.linspace(0,image.shape[1]-1,image.shape[1])
    Z=np.linspace(0,image.shape[2]-1,image.shape[2])
    #a,b=np.meshgrid(Y,Z)
    c,d=np.meshgrid(X,Y)
    #e,f=np.meshgrid(X,Z)
    cest = ax.contourf(c,d, np.transpose(mesh_z), zdir='z', offset=0, cmap="Blues")
    #cest = ax.contourf(np.transpose(mesh_x),b,a,zdir='x', offset=0, cmap="Greys")
    #cest = ax.contourf(e,np.transpose(mesh_y),f,zdir="y", offset=image.shape[1], cmap="Greys")
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xlim(0, image.shape[0])
    ax.set_ylim(0, image.shape[1])
    ax.set_zlim(0, image.shape[2])
    ax.set_title(name+"_3D_Complete.png")
    if save:
        fig.savefig("/home/nnk/Bachelor/LiTS/combined_vol_seg/"+name+"_3D_Complete.png", bbox_inches='tight')
    plt.close("all")
    del mesh, mesh_vol, face_color, face_color_vol
    if check:
        del mesh2, face_color2, verts2, faces2
    del verts,verts_vol,faces,faces_vol
    gc.collect()
#Get majority label in image
def largest_label_volume(img, bg=0):
    vals, counts = np.unique(img, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def make_hist(img):
    plt.hist(img.ravel(),bins=80)
    plt.xlabel("normalized Hounsfield Units (nHU)")
    plt.ylabel("Frequency")
    plt.title("Minimum bound: -1000 (Air), Maximum bound: 400 (Bones)")
    plt.show()
#3D plot all available segmentation images
def make_all_liver_images():
    total_n = len(segmentations)
    mean_t = 0
    for i in xrange(len(segmentations)):
        start = time.time()
        plot_data=np.asarray(segmentation_list[i].dataobj)
        plot_3d_seg(plot_data,segmentations[i].split(".")[0],1,True)
        end   = time.time()
        el_t  = end-start
        mean_t += el_t
        pred_t = (total_n-i)*(mean_t/(i+1))
        print "Plotted: {}/{} in {}s.".format(i+1,total_n,el_t)
        print "To Go: {}s.".format(pred_t)
        del plot_data

#im_seg = np.array(segmentation_list[8].get_data())
#plot_3d_seg(im_seg,"test",1,False)
#plot_3d_all(im_vol, im_seg,"Testname",threshold_bones=350,save=False)
#make_all_liver_images()
#plt.close("all")
from scipy import misc as mc

#Check overlay
#im_vol=np.array(volume_list[51].get_data())
#im_seg=np.array(segmentation_list[51].get_data())
for i in xrange(im_vol.shape[2]):
    plt.imshow(im_vol[:,:,i],cmap="inferno")
    plt.imshow(im_seg[:,:,i],cmap="Greys",alpha=0.35)
    plt.show()

#Load resampled data:
INPUT_FOLDER = "/media/nnk/Z-NochMehrPlatz/LiTS_data/Resampled"
data = os.listdir(INPUT_FOLDER)
data.sort()
del data[0]
split=len(data)/2
segmentations_temp = data[0:split]
volumes_temp = data[split:]
volumes = [INPUT_FOLDER+"/"+i for i in volumes_temp]
segmentations = [INPUT_FOLDER+"/"+i for i in segmentations_temp]

#Combined 3D plot
sudoPassword = 'dl3163745'
command = 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches'
mean_t=0
jp=96
total_n = len(segmentations)-jp

for k in xrange(total_n):
    im_vol=np.transpose(np.load(volumes[k+jp]),(1,0,2))
    im_seg=np.transpose(np.load(segmentations[k+jp]),(1,0,2))
    im_vol.shape
    im_seg.shape
    np.unique(im_vol)
    np.unique(im_seg)
    start = time.time()
    print "Working on 3D-segmentation..."
    plot_3d_seg(im_seg,"Segmentation_"+str(k+jp),save=True)
    p = os.system('echo %s|sudo -S %s' % (sudoPassword, command))
    print "Working on 3D-Volume..."
    plot_3d_vol(im_vol,"Volume_"+str(k+jp),save=True)
    p = os.system('echo %s|sudo -S %s' % (sudoPassword, command))
    #for i in xrange(im_vol.shape[2]):
    #    to_rot = im_vol[:,:,i]
    #    im_vol[:,:,i]=np.rot90(np.rot90(to_rot))
    print "Working on 3D-Combination..."
    plot_3d_all(im_vol, im_seg,"Volume_Segmentation"+str(k+jp),threshold_bones=320,save=True)
    plt.close("all")
    del im_vol,im_seg
    gc.collect()

    p = os.system('echo %s|sudo -S %s' % (sudoPassword, command))
    end = time.time()
    el_t = end-start
    mean_t += el_t
    pred_t = (total_n-k)*(mean_t/(k+1))
    print "Finished plotting 3D image: ",k+1
    print "This took: {}s, projected time: {}s.".format(end-start,pred_t)



""""""""""""""""""""""""""""""""""""""""""""""""
""" Check out HU-liver mask """
""""""""""""""""""""""""""""""""""""""""""""""""
#Issue: overlap with muscle and blood/slight difference for different CT scans
liver_lower_bound=0.
liver_upper_bound=100.

#Use HU-boundaries to create liver mask
def get_HU_liver_mask(img,lower,upper):
        img = img.copy() #Not to alter value
        img[img>upper]=0
        img[img<lower]=0
        img[img!=0]=1
        return img
        del img

#Check average HU-values in provided liver-segmentation
def get_HU_dist_in_liver_seg(imgm,segm,check=False):
    seg_liver = np.zeros_like(segm)*1.
    if not check:
        check = np.max(np.unique(segm))>1
    if check:
        seg_nodules = np.zeros_like(segm)*1.
        seg_nodules[segm==2]=1.
    else:
        print "No nodules existent."
    seg_liver[segm==1]=1.
    liver_segment=imgm*seg_liver
    if check:
        nodule_segment=imgm*seg_nodules
        return liver_segment,nodule_segment
    else:
        return liver_segment
    del liver_segment,nodule_segment,seg_nodules,seg_liver

def get_seg_dist(seg_l,seg_n=None,shape=[2,2,2],check=True,bound_l=1,bound_n=2,name="None",save=False,plot=True):
    cts_l = 0.
    cts_n = 0.
    for i in xrange(shape.shape[2]):
        if bound_l in img_seg[:,:,i]:
            cts_l+=1.
        if bound_n in img_seg[:,:,i]:
            cts_n+=1.
    k_l=np.sum(seg_l,axis=2)/cts_l
    lo_l= filter(lambda x : x != 0., k_l.flatten())
    if check:
        k_n=np.sum(seg_n,axis=2)/cts_n
        lo_n= filter(lambda x : x != 0., k_n.flatten())
    if check:
        f,ax = plt.subplots(1,2)
        _,_,_=ax[0].hist(lo_l,bins=80);
        ax[0].set_xlabel("Houndsfield Unit (HU)")
        ax[0].set_ylabel("Frequency")
        ax[0].set_title("HU-dist. in liver")
        _,_,_=ax[1].hist(lo_n,bins=80);
        ax[1].set_xlabel("Houndsfield Unit (HU)")
        ax[1].set_ylabel("Frequency")
        ax[1].set_title("HU-dist. in liver-nodules")
        if name!="None":
            f.suptitle("Liver/nodule histograms for "+name)
        else:
            f.suptitle("HU-dist. in liver/nodule segmentation")
            name="Check"
        f.tight_layout()
        f.subplots_adjust(top=0.8)
    else:
        f,ax = plt.subplots(1,1)
        _,_,_=ax.hist(lo_l,bins=60);
        ax.set_xlabel("Houndsfield Unit (HU)")
        ax.set_ylabel("Frequency")
        ax.set_title("Histogram for HU-dist. in liver, no nodules")
    if save:
        ts = "hist_seg_dist_"+name+".png"
        f.savefig(ts,bbox_inches="tight")
    if plot:
        plt.show()
    else:
        plt.close(f)
    gc.collect()
    #plt.xlim([0,100])

HU_liver_mask = get_HU_liver_mask(img_vol,liver_lower_bound,liver_upper_bound)
for i in xrange(HU_liver_mask.shape[2]):
    plt.imshow(HU_liver_mask[:,:,i],cmap="Greys")
    plt.show()

for nift in xrange(len(volume_list)):
    img_vol = np.asarray(volume_list[nift].dataobj)
    img_seg = np.asarray(segmentation_list[nift].dataobj)
    check = np.max(np.unique(img_seg))>1
    if check:
        liver_seg,nodule_seg=get_HU_dist_in_liver_seg(img_vol,img_seg,check=check)
    else:
        liver_seg=get_HU_dist_in_liver_seg(img_vol,img_seg)
    get_seg_dist(liver_seg,nodule_seg,shape=img_seg,check=check,name=volumes[nift].split(".")[0],save=True,plot=False)
    plt.close("all")
    del img_vol, img_seg, liver_seg, nodule_seg
    gc.collect()
    print "Computed hist. for ",volumes[nift]

""""""""""""""""""""""""""""""""""""""""""""""""
""" Compute Pixel Mean """
""""""""""""""""""""""""""""""""""""""""""""""""
mean_vals = []
mean_change=0
mean_changes=[]
for i in xrange(len(volumes)):
    mm = np.mean(normalize(np.load(volumes[i])))
    mean_vals.append(mm)
    mean_change+=mm
    mean_changes.append(mean_change/(i+1))
    print "computed pixed_mean for image {}/{} with value: {}".format(i,len(volumes),mm)
    del mm
    gc.collect()
    p = os.system('echo %s|sudo -S %s' % (sudoPassword, command))
PIXEL_MEAN=np.mean(mean_vals)
f,ax = plt.subplots(1,1)
ax.plot(np.linspace(1,len(volumes),len(volumes)),mean_changes)
ax.set_title("Behaviour of average pixel mean, final value: "+str(PIXEL_MEAN))
ax.set_xlabel("Volume number")
ax.set_ylabel("Pixel mean")
f.savefig("pix_mean_conv.png",bbox_inches="tight")
plt.show()

""""""""""""""""""""""""""""""""""""""""""""""""
""" Load and Prepare the data """
""""""""""""""""""""""""""""""""""""""""""""""""
MIN_BOUND = -1000.0 #Everything below: Water
MAX_BOUND = 400.0 #Everything above corresponds to bones
PIXEL_MEAN = 0.338 #See above, for normalized training data

def set_bounds(image):
    image[image>MAX_BOUND]=MAX_BOUND
    image[image<MIN_BOUND]=MIN_BOUND
    return image
def normalize(image,zero_center=False):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    if zero_center:
        image = image - PIXEL_MEAN
    return image

#Convert to 5dim-shape
def load_and_prepare(to_prep,wlist=None):
    data_stack = []
    if wlist == None:
        for nift in to_prep:
            to_app=np.expand_dims(np.transpose(normalize(np.array(nift.dataobj),zero_center=True),(2,0,1)),axis=3)
            data_stack.append(np.expand_dims(to_app,axis=4))
    else:
        for choice in wlist:
            to_app=np.expand_dims(np.transpose(normalize(np.array(to_prep[choice].dataobj),zero_center=True),(2,0,1)),axis=3)
            data_stack.append(np.expand_dims(to_app,axis=4))
    data_stack = np.stack(data_stack,axis=0)
    return data_stack
    del data_stack,to_app

def resample(img, seg, scan, new_voxel_dim=[1,1,1]):
    # Get voxel size
    voxel_dim = np.array(scan.header.structarr["pixdim"][1:4],dtype=np.float32)
    # Resample to optimal [1,1,1] voxel size
    resize_factor = voxel_dim / new_voxel_dim
    scan_shape=np.array(scan.header.get_data_shape())
    new_scan_shape = scan_shape * resize_factor
    rounded_new_scan_shape = np.round(new_scan_shape)
    rounded_resize_factor = rounded_new_scan_shape / scan_shape # Change resizing due to round off error
    new_voxel_dim = voxel_dim / rounded_resize_factor

    img = ndi.interpolation.zoom(img, rounded_resize_factor, mode='nearest')
    seg = ndi.interpolation.zoom(seg, rounded_resize_factor, mode='nearest')
    return img, seg, new_voxel_dim
    del img,seg

spacings=[]
#Resample all Scans
INPUT_FOLDER_res = "/media/nnk/Z-NochMehrPlatz/LiTS_data/Resampled"
sudoPassword = 'dl3163745'
command = 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches'
jp=0

for i in xrange(len(volume_list)-jp):
    img_vol = np.asarray(volume_list[i+jp].dataobj).astype(np.int32)
    img_seg = np.asarray(segmentation_list[i+jp].dataobj).astype(np.int32)
    volume_list[i+jp].uncache()
    segmentation_list[+jp].uncache()
    print "Resampling..."
    try:
        new_vol,new_seg,new_vdim=resample(img_vol,img_seg,volume_list[i+jp])
    except ValueError:
        print "Error at file ",volumes[i]
        new_vol,new_seg,new_vdim=resample(img_vol,img_seg,volume_list[i+jp])
    print "Saving..."
    #np.savez_compressed(INPUT_FOLDER+"/Resampled/"+volumes[i].split(".")[0],new_vol)
    np.save(INPUT_FOLDER_res+"/"+volumes[i+jp].split(".")[0],new_vol)
    np.save(INPUT_FOLDER_res+"/"+segmentations[i+jp].split(".")[0],new_seg)
    spacings.append(new_vdim)
    del img_vol, img_seg, new_vol, new_seg, new_vdim
    gc.collect()
    p = os.system('echo %s|sudo -S %s' % (sudoPassword, command))
    print "Finished resampling and saving ",volumes[i+jp]
    print "Progress: {}/{}.".format(i+1,len(volume_list)-jp)
np.save(INPUT_FOLDER_res+"/new_spacings",spacings)

for i in xrange(len(volumes)):
    print i,volumes[i]
# To load npz: kk = np.load("/media/nnk/Z-NochMehrPlatz/Bachelor/LiTS/test.npz"), p = kk.f.arr_0

PRELOAD = False
if PRELOAD:
    CT_data = load_and_prepare(volume_list)
    labels  = load_and_prepare(segmentation_list)

img_vol=np.array(volume_list[34].dataobj)
img_res,new_v=resample(img_vol,volume_list[34])



""""""""""""""""""""""""""""""""""""""""""""""""
""" Load Neural Net """
""""""""""""""""""""""""""""""""""""""""""""""""
reload(network)
net_output_layer,imageout = network.construct_unet_3D(channels=1, no_f_base=8, f_size=3, branches=[2,2,2,2], dropout=0,
                                                      bs=None, class_nums=2, pad="same", nonlinearity=lasagne.nonlinearities.rectify,
                                                      input_dim=[None,None,None])
print("#Layers: %d, #Parameters (Trainable/complete parameter count): %d/%d ."%(sum(hasattr(layer, 'W') for layer in lasagne.layers.get_all_layers(net_output_layer)),
lasagne.layers.count_params(net_output_layer, trainable=True),lasagne.layers.count_params(net_output_layer)))
print "---------------------------------"



""""""""""""""""""""""""""""""""""""""""""""""""
""" Compile all theano functions """
""""""""""""""""""""""""""""""""""""""""""""""""
#Necessary symbolic variables
print "Start compilation..."
tensor5d = theano.tensor.TensorType('float32', (False,)*5)
input_var = tensor5d("input_var")
segmented_gt_vector = theano.tensor.ivector()

#Define hyperparameters. These could also be symbolic variables
lr = theano.shared(np.float32(0.0005)) #Shared bc. we decrease it while training.
weight_decay = 1e-4
momentum = 0.95

#L2-regularization
l2_reg_correction = lasagne.regularization.regularize_network_params(net_output_layer, lasagne.regularization.l2) * weight_decay

#If dropout is enable, we have to distinguish between predictions for test and train sets.
prediction_train = lasagne.layers.get_output(net_output_layer, input_var, deterministic=False)
prediction_test = lasagne.layers.get_output(net_output_layer, input_var, deterministic=True)
#get_pred_tr2 = theano.function([input_var],prediction_train)
#get_pred_bla2 = theano.function([input_var,segmented_gt_vector],lasagne.objectives.categorical_crossentropy(prediction_train, segmented_gt_vector))

prediction_image = lasagne.layers.get_output(imageout,input_var,deterministic = True) #For U-Net
#prediction_image = lasagne.layers.get_output(image_out,input_var,deterministic = True)

#Final Loss function (we assume that input is flattened to 2D):
loss_train = lasagne.objectives.categorical_crossentropy(prediction_train, segmented_gt_vector)
loss_train = loss_train.mean()
loss_train += l2_reg_correction #L2 regularization
#----------
loss_test = lasagne.objectives.categorical_crossentropy(prediction_test, segmented_gt_vector)
loss_test = loss_test.mean()
loss_test += l2_reg_correction #L2 regularization

#Compute training accuracy
training_acc  = T.mean(T.eq(T.argmax(prediction_train,axis=1), segmented_gt_vector), dtype=theano.config.floatX)
test_acc= T.mean(T.eq(T.argmax(prediction_test,axis=1), segmented_gt_vector), dtype=theano.config.floatX)
#training_acc  = T.mean(T.eq(prediction_train, segmented_gt_vector), dtype=theano.config.floatX)
#test_acc= T.mean(T.eq(prediction_test, segmented_gt_vector), dtype=theano.config.floatX)

#updates
net_params = lasagne.layers.get_all_params(net_output_layer, trainable=True)
#updates = lasagne.updates.nesterov_momentum(loss_train, net_params, lr, momentum)

beta1=0.9
beta2=0.999
epsilon=1e-8
updates= lasagne.updates.adam(loss_train, net_params, lr, beta1, beta2, epsilon)

#Training functions
train_fn = theano.function([input_var, segmented_gt_vector], [loss_train, training_acc], updates=updates)
print "1/6 functions compiled."
val_fn = theano.function([input_var, segmented_gt_vector], [loss_test, test_acc])
print "2/6 functions compiled."

#Make predictions for test sets:
make_prediction_flat = theano.function([input_var], prediction_test)
print "3/6 functions compiled."
make_prediction_image= theano.function([input_var],prediction_image)
print "4/6 functions compiled."
#Add image weighting and data augmentation.

grad_loss_train = T.grad(loss_train,net_params)
gradient_check = theano.function([input_var, segmented_gt_vector],grad_loss_train)
print "5/6 functions compiled."
loss_train_func = theano.function([input_var, segmented_gt_vector],loss_train)
print "6/6 functions compiled."

print "Compiling done."
print "------------------------------"
