"""
Set everything up to uploadable data
"""

""""""""""""""""""""""""""""""""""""""""""""""""
""" Load necessary libs """
""""""""""""""""""""""""""""""""""""""""""""""""
import cPickle
import numpy as np
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
import gc
import own_layers
from scipy import ndimage as ndi
import LiTS_helper_functions as lhf
from PIL import Image
from PIL import ImageEnhance
from scipy import misc as mc


""""""""""""""""""""""""""""""""""""""""""""""""
""" Load Test Data """
""""""""""""""""""""""""""""""""""""""""""""""""
os.chdir("/home/nnk/Bachelor/LiTS") #Working directory
INPUT_FOLDER = "/media/nnk/Z-NochMehrPlatz/LiTS_data/Test_data" #Path to resampled data

volumes_temp = os.listdir(INPUT_FOLDER)
volumes_temp.sort()

volumes = [INPUT_FOLDER+"/"+i for i in volumes_temp]

sudoPassword = 'sudopw'
command = 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches'



""""""""""""""""""""""""""""""""""""""""""""""""
""" Load and initialize nets with weights """
""""""""""""""""""""""""""""""""""""""""""""""""
reload(network)
f_size_l=3
Liver_net,Liver_image_net = network.construct_unet_recursive(channels=1, no_f_base=32, f_size=f_size_l, branches=[2,2,2,2], dropout=0.3,
                                                      bs=None, class_nums=2, pad="same", nonlinearity=lasagne.nonlinearities.rectify,
                                                      input_dim=[None,None])
print "Network statistics:"
print("#Layers: %d\n#Parameters (Trainable/complete parameter count): %d/%d ."%(sum(hasattr(layer, 'W') for layer in lasagne.layers.get_all_layers(Liver_net)),
lasagne.layers.count_params(Liver_net, trainable=True),lasagne.layers.count_params(Liver_net)))
print "---------------------------------"
####
f_size_n=3
Nodule_net,Nodule_image_net = network.construct_unet_recursive(channels=1, no_f_base=32, f_size=f_size_n, branches=[2,2,2,2], dropout=0.3,
                                                      bs=None, class_nums=2, pad="same", nonlinearity=lasagne.nonlinearities.rectify,
                                                      input_dim=[None,None])
print "Network statistics:"
print("#Layers: %d\n#Parameters (Trainable/complete parameter count): %d/%d ."%(sum(hasattr(layer, 'W') for layer in lasagne.layers.get_all_layers(Nodule_net)),
lasagne.layers.count_params(Nodule_net, trainable=True),lasagne.layers.count_params(Nodule_net)))
print "---------------------------------"


input_var = theano.tensor.tensor4('input_var')
segmented_gt_vector = theano.tensor.ivector()
L_prediction_image = lasagne.layers.get_output(Liver_image_net,input_var,deterministic = True) #For U-Net
N_prediction_image = lasagne.layers.get_output(Nodule_image_net,input_var,deterministic = True) #For U-Net
Liver_prediction_image= theano.function([input_var], L_prediction_image)
Nodule_prediction_image= theano.function([input_var], N_prediction_image)

with open("UNet_2D_fb32_fs3_33333_d02_LiverOnly_downsampled_parameters.pkl", 'r') as f:
    Liver_net_params = cPickle.load(f)
with open("UNet_2D_fb32_fs3_33333_d02_NoduleOnly_parameters_by_end_lr1p59eM6.pkl", 'r') as f:
    Nodule_net_params = cPickle.load(f)
lasagne.layers.set_all_param_values(Liver_net, Liver_net_params)
lasagne.layers.set_all_param_values(Nodule_net, Nodule_net_params)



""""""""""""""""""""""""""""""""""""""""""""""""
""" Re- and Downsample Testdata """
""""""""""""""""""""""""""""""""""""""""""""""""
#optional, I trained my net on re- and downsampled data, so I need to do the same for the test set.

resample_data = False
if resample_data:
    INPUT_FOLDER_space="/media/nnk/Z-NochMehrPlatz/LiTS_data"
    INPUT_FOLDER_down = "/media/nnk/Z-NochMehrPlatz/LiTS_data/Downsampled_Test"

    def resample_test(img, scan, new_voxel_dim=[1,1,1]):
        # Get voxel size
        scan = nib.load(scan)
        voxel_dim = np.array(scan.header.structarr["pixdim"][1:4],dtype=np.float32)
        # Resample to optimal [1,1,1] voxel size
        resize_factor = voxel_dim / new_voxel_dim
        scan_shape=np.array(scan.header.get_data_shape())
        new_scan_shape = scan_shape * resize_factor
        rounded_new_scan_shape = np.round(new_scan_shape)
        rounded_resize_factor = rounded_new_scan_shape / scan_shape # Change resizing due to round off error
        new_voxel_dim = voxel_dim / rounded_resize_factor
        img = ndi.interpolation.zoom(img, rounded_resize_factor, mode='nearest',order=1)

        return img,new_voxel_dim
        del img

    def downsample(img, new_voxel_dim=[1,1,1]):
        # Get voxel size
        voxel_dim = np.array([1,1,1])
        # Resample to optimal [1,1,1] voxel size
        resize_factor = voxel_dim / new_voxel_dim
        scan_shape=np.array(img.shape)
        new_scan_shape = scan_shape * resize_factor
        rounded_new_scan_shape = np.round(new_scan_shape)
        rounded_resize_factor = rounded_new_scan_shape / scan_shape # Change resizing due to round off error
        new_voxel_dim = voxel_dim / rounded_resize_factor
        img = ndi.interpolation.zoom(img, rounded_resize_factor, mode='nearest',order=1)

        return img, new_voxel_dim
        del img

    spacings_down=[]
    spacings_res=[]
    jp=0 #to start from any index I want, not really necessary
    for i in xrange(len(volumes)-jp):
        img_vol = np.asarray(nib.load(volumes[i+jp]).dataobj).astype(np.int32)
        print "Resampling..."
        new_vol_res,new_vdim_res=resample_test(img_vol,volumes[i+jp])
        new_vol_down,new_vdim_down=downsample(new_vol_res,[2.,2.,1.])

        print "Saving..."
        #np.savez_compressed(INPUT_FOLDER+"/Resampled/"+volumes[i].split(".")[0],new_vol)
        np.save(INPUT_FOLDER_down+"/"+volumes_temp[i+jp].split(".")[0],new_vol_down)
        spacings_down.append(new_vdim_res)
        spacings_res.append(new_vdim_down)

        print "Finished resampling and saving ",volumes_temp[i+jp]
        print "Progress: {}/{}.".format(i+1,len(volumes)-jp)

        del img_vol, new_vol_res, new_vol_down
        gc.collect()
        p = os.system('echo %s|sudo -S %s' % (sudoPassword, command))

    np.save(INPUT_FOLDER_space+"/new_test_spacings_down",spacings_down)
    np.save(INPUT_FOLDER_space+"/new_test_spacings_res",spacings_res)


""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Run the nodule segmentation on the test data """
""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Function to get segmentation from probability map, works also for non-normalized maps
def make_maps(predicts,do_enhance=False):
    prob_map = np.zeros_like(predicts[:,0:1,:,:])
    seg_map  = np.zeros_like(predicts[:,0:1,:,:])
    for i in xrange(prob_map.shape[0]):
        p_map_temp_1 = (predicts[i,0:1,:,:]-np.min(predicts[i,0:1,:,:]))/(np.max(predicts[i,0:1,:,:])-np.min(predicts[i,0:1,:,:]))
        p_map_temp_2 = (predicts[i,1:2,:,:]-np.min(predicts[i,1:2,:,:]))/(np.max(predicts[i,1:2,:,:])-np.min(predicts[i,1:2,:,:]))
        p_map_temp   = (p_map_temp_2+1-p_map_temp_1)/2
        if do_enhance:
            im = Image.fromarray(np.uint8(p_map_temp[0,:,:]*255))
            k2 = ImageEnhance.Contrast(im)
            kl = k2.enhance(3.0)
            new_seg=np.array(kl)
            new_seg=new_seg/float(np.max(new_seg))
            p2 = (new_seg)>0.9
            seg_map[i,0,:,:] = p2
        else:
            seg_map[i,0,:,:] = p_map_temp[0,:,:]>0.8
        prob_map[i,0,:,:]= p_map_temp[0,:,:]
    return prob_map,seg_map


#Do the actual lesion segmentations for the test data
SAVE_FOLDER="/media/nnk/Z-NochMehrPlatz/LiTS_data/Test_predictions"
INPUT_FOLDER_p = "/media/nnk/Z-NochMehrPlatz/LiTS_data/Downsampled_Test" #Path to resampled data
volumes_temp_p = os.listdir(INPUT_FOLDER_p)
volumes_temp_p.sort()
volumes_p = [INPUT_FOLDER_p+"/"+i for i in volumes_temp_p]

for testid in xrange(len(volumes_p)):
    testid=0
    test_data = lhf.normalize(lhf.fit_dimensions(np.load(volumes_p[testid])[:,:,:]),use_bd=True).astype(np.float32)
    preds=[]
    for i in xrange(test_data.shape[2]):
        p_temp=Liver_prediction_image(test_data[:,:,i,:,:])
        p_temp[0,0,:,:]=(p_temp[0,0,:,:]-np.min(p_temp[0,0,:,:]))/(np.max(p_temp[0,0,:,:])-np.min(p_temp[0,0,:,:]))
        p_temp[0,1,:,:]=(p_temp[0,1,:,:]-np.min(p_temp[0,1,:,:]))/(np.max(p_temp[0,1,:,:])-np.min(p_temp[0,1,:,:]))
        preds.append(p_temp)
    preds=np.vstack(preds)
    _,liver_mask = make_maps(preds)
    """liver_mask.shape
    for i in xrange(preds.shape[0]):
        f,ax = plt.subplots(1,2)
        ax[0].imshow(test_data[0,0,i,:,:],cmap="gray_r")
        ax[0].imshow(liver_mask[i,0,:,:],cmap="inferno",alpha=0.3)
        ax[1].imshow(liver_mask[i,0,:,:],cmap="gray_r")
        plt.show()"""
    preds=[]
    for i in xrange(test_data.shape[2]):
        to_nod_pred = test_data[:,:,i,:,:]
        to_nod_pred[liver_mask[i:i+1,:,:,:]==0]=0. #mask liver-only data
        p_temp=Nodule_prediction_image(to_nod_pred)
        p_temp[0,0,:,:]=(p_temp[0,0,:,:]-np.min(p_temp[0,0,:,:]))/(np.max(p_temp[0,0,:,:])-np.min(p_temp[0,0,:,:]))
        p_temp[0,1,:,:]=(p_temp[0,1,:,:]-np.min(p_temp[0,1,:,:]))/(np.max(p_temp[0,1,:,:])-np.min(p_temp[0,1,:,:]))
        preds.append(p_temp)
    preds=np.vstack(preds)
    _,nodule_mask = make_maps(preds)
    """for i in xrange(preds.shape[0]):
        f,ax = plt.subplots(1,2)
        ax[0].imshow(test_data[0,0,i,:,:],cmap="gray_r")
        ax[0].imshow(nodule_mask[i,0,:,:],cmap="inferno",alpha=0.3)
        ax[1].imshow(nodule_mask[i,0,:,:],cmap="gray_r")
        plt.show()"""
    #save data just in case something goes wrong in the Nifti-step
    np.save(SAVE_FOLDER+"/"+volumes_temp_p[testid].split(".")[0],nodule_mask[:,0,:,:].transpose(1,2,0))


""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Up- and Backsample the data, then convert to Nifti """
""""""""""""""""""""""""""""""""""""""""""""""""""""""
#I have to do this since I re- and downsampled the data
INPUT_FOLDER_t = "/media/nnk/Z-NochMehrPlatz/LiTS_data"
resample_spacing = np.load(INPUT_FOLDER_t+"/new_test_spacings_res.npy")
down_spacing = np.load(INPUT_FOLDER_t+"/new_test_spacings_down.npy")

SAVE_FOLDER_up="/media/nnk/Z-NochMehrPlatz/LiTS_data/to_upload"
INPUT_FOLDER_up = "/media/nnk/Z-NochMehrPlatz/LiTS_data/Test_predictions" #Path to resampled data
volumes_temp_up = os.listdir(INPUT_FOLDER_up)
volumes_temp_up.sort()
volumes_up = [INPUT_FOLDER_up+"/"+i for i in volumes_temp_up]

def upsample(img, original=None, rfactor=[0.5,0.5,1.], current_voxel_dim=[1.,1.,1.]):
    current_voxel_dim=np.array(current_voxel_dim)
    rfactor          =np.array(rfactor)
    if original:
        scan = nib.load(original)
        refactor=np.array(scan.header.get_data_shape())/np.array(img.shape,dtype=np.float32)
        original_voxel_dim = np.array(scan.header.structarr["pixdim"][1:4],dtype=np.float32)
        new_resize_factor = original_voxel_dim/current_voxel_dim
        img = ndi.interpolation.zoom(img,refactor, mode="nearest", order=1)
    else:
        new_resize_factor = current_voxel_dim/rfactor
        img = ndi.interpolation.zoom(img, new_resize_factor, mode="nearest", order=1)
    return img

#Part where I upsample and also convert to Nifti. Don't forget to set
#the nodule labels from 1 to 2 (at least I have to :)).
for vup in xrange(len(volumes_up)):
    test_up    = np.load(volumes_up[vup])
    test_up    =test_up.astype(np.int8)
    up_by_2    = upsample(test_up,rfactor=[0.5,0.5,1.],current_voxel_dim=down_spacing[vup])
    up_to_orig = upsample(up_by_2,original=volumes[0],current_voxel_dim=down_spacing[vup])
    up_to_orig[up_to_orig==1]=2
    up_to_orig = up_to_orig.astype(np.int8)
    #Save Nifti data with correct naming
    nib.save(nib.Nifti1Image(up_to_orig,np.eye(4)),SAVE_FOLDER_up+"/test-segmentation-"+volumes_temp[vup].split("-")[2].split(".")[0]+".nii")
