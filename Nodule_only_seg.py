"""
Run Neural Net on prepared data
"""

""""""""""""""""""""""""
""" Load Libraries """
""""""""""""""""""""""""
import cPickle
import numpy as np
import os
import time
import theano
from theano import tensor as T
from scipy import misc as mc
import lasagne
from lasagne.layers import concat,InputLayer, ConcatLayer, Pool2DLayer, Deconv2DLayer, Conv2DLayer, DenseLayer
from lasagne.layers import ReshapeLayer, DimshuffleLayer, NonlinearityLayer, DropoutLayer
import network_library_functions as network
import nibabel as nib
import LiTS_helper_functions as lhf
import theano.sandbox.cuda.basic_ops as sbcuda
from scipy import misc as mc
from scipy import ndimage as ndi
import matplotlib.pyplot as plt


""""""""""""""""""""""""""""""""""""""""""""""""
""" Set necessary paths/load data paths """
""""""""""""""""""""""""""""""""""""""""""""""""
os.chdir("/home/nnk/Bachelor/LiTS") #Working directory
INPUT_FOLDER = "/media/nnk/Z-NochMehrPlatz/LiTS_data/Downsampled_Volume" #Path to resampled data
INPUT_FOLDER2= "/media/nnk/Z-NochMehrPlatz/LiTS_data/Downsampled_NoduleMask"
INPUT_FOLDER3= "/media/nnk/Z-NochMehrPlatz/LiTS_data/Downsampled_LiverMask"


volumes_temp = os.listdir(INPUT_FOLDER)
volumes_temp.sort()
segmentations_temp = os.listdir(INPUT_FOLDER2)
segmentations_temp.sort()
liver_segs = os.listdir(INPUT_FOLDER3)
liver_segs.sort()

volumes = [INPUT_FOLDER+"/"+i for i in volumes_temp]
segmentations = [INPUT_FOLDER2+"/"+i for i in segmentations_temp]
livers = [INPUT_FOLDER3+"/"+i for i in liver_segs]

sudoPassword = 'sudopw'
command = 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches'

""""""""""""""""""""""""""""""""""""""""""""""""
""" Load Neural Net """
""""""""""""""""""""""""""""""""""""""""""""""""
reload(network)
General_name="UNet_2D_fb32_fs3_33333_d03_NoduleOnly"
f_size=3
net_output_layer,imageout = network.construct_unet_recursive(channels=1, no_f_base=32, f_size=f_size, branches=[2,2,2,2], dropout=0.2,
                                                      bs=None, class_nums=2, pad="same", nonlinearity=lasagne.nonlinearities.rectify,
                                                      input_dim=[None,None])
print "Network statistics:"
print "Name: ",General_name
print("#Layers: %d\n#Parameters (Trainable/complete parameter count): %d/%d ."%(sum(hasattr(layer, 'W') for layer in lasagne.layers.get_all_layers(net_output_layer)),
lasagne.layers.count_params(net_output_layer, trainable=True),lasagne.layers.count_params(net_output_layer)))
print "---------------------------------"


""""""""""""""""""""""""""""""""""""""""""""""""
""" Augment data """
""""""""""""""""""""""""""""""""""""""""""""""""
def augment_data(img,mode=["flip","rescale"],seeed=111,to_type=np.int32,rand_choice=True):
    np.random.seed(seeed)
    t_rot=np.random.randint(-180,180,1)
    np.random.seed(seeed)
    resc =np.random.uniform(0.8,1.2,1)
    augmentation=np.zeros_like(img)

    perform_rot =True
    perform_dist=True
    perform_resc=True
    perform_flip=True

    if rand_choice:
        np.random.seed(seeed)
        if np.round(np.random.uniform(0,1))>0.5:
            perform_rot=False
        np.random.seed(seeed)
        if np.round(np.random.uniform(0,1))>0.5:
            perform_dist=False
        np.random.seed(seeed)
        if np.round(np.random.uniform(0,1))>0.5:
            perform_resc=False
        np.random.seed(seeed)
        if np.round(np.random.uniform(0,1))>0.5:
            perform_flip=False

    do_one=[perform_rot,perform_dist,perform_resc,perform_flip]
    if True not in do_one:
        perform_flip=True

    if "rot" in mode and perform_rot:
        for i in xrange(img.shape[2]):
            temp_save=ndi.interpolation.rotate(img[0,0,i,:,:],t_rot,order=1)
            temp_save[temp_save!=0]=1.
            augmentation[0,0,i,:,:]=temp_save.astype(to_type)

    if "distort" in mode and perform_dist:
        print "Not implemented yet!"

    if "rescale" in mode and perform_resc:
        for i in xrange(img.shape[2]):
            augmentation[0,0,i,:,:]=ndi.interpolation.zoom(img[0,0,i,:,:],resc,order=1)

    if "flip" in mode and perform_flip:
        for i in xrange(img.shape[2]):
            augmentation[0,0,i,:,:]=np.fliplr(img[0,0,i,:,:])

    return augmentation


""""""""""""""""""""""""""""""""""""""""""""""""
""" Convert to liver-only slices """
""""""""""""""""""""""""""""""""""""""""""""""""
def do_liver_only_slices(img,seg,seg_check):
    img_coll=[]
    seg_coll=[]
    seg_check_coll=[]
    for i in xrange(img.shape[2]):
        if 1 in seg_check[:,:,i,:,:] or 2 in seg[:,:,i,:,:]:
            img_coll.append(img[:,:,i,:,:])
            seg_coll.append(seg[:,:,i,:,:])
            seg_check_coll.append(seg_check[:,:,i,:,:])
    return np.stack(img_coll,axis=2),np.stack(seg_coll,axis=2),np.stack(seg_check_coll,axis=2)


""""""""""""""""""""""""""""""""""""""""""""""""
""" Crop liver to train """
""""""""""""""""""""""""""""""""""""""""""""""""
#Randomly crop 2D-patches from 2D-data
def crop_2D(img,seg,liver_check,no_crops=10,crop_size=[128,128],seeed=111,only_in_liver=False):
    crop_collect_img=[]
    crop_collect_seg=[]
    np.random.seed(seeed)
    liver_check=liver_check[0,0,:,:]
    liver_coords = np.array(np.where(liver_check==1)).transpose()
    nodule_coords= np.array(np.where(seg[0,0,:,:]==1)).transpose()
    nod_count = len(nodule_coords)
    for i in xrange(no_crops):
        if only_in_liver:
            if nod_count>0:
                x = np.random.randint(0,nodule_coords.shape[0])
                to_app_vol = img[0:1,0:1,np.clip(nodule_coords[x,0]-crop_size[0]/2,0,None):nodule_coords[x,0]+crop_size[0]/2,np.clip(nodule_coords[x,1]-crop_size[1]/2,0,None):nodule_coords[x,1]+crop_size[1]/2]
                to_app_seg = seg[0:1,0:1,np.clip(nodule_coords[x,0]-crop_size[0]/2,0,None):nodule_coords[x,0]+crop_size[0]/2,np.clip(nodule_coords[x,1]-crop_size[1]/2,0,None):nodule_coords[x,1]+crop_size[1]/2]
                crop_collect_img.append(to_app_vol)
                crop_collect_seg.append(to_app_seg)
                nod_count-=1
            else:
                x = np.random.randint(0,liver_coords.shape[0])
                to_app_vol = img[0:1,0:1,np.clip(liver_coords[x,0]-crop_size[0]/2,0,None):liver_coords[x,0]+crop_size[0]/2,np.clip(liver_coords[x,1]-crop_size[1]/2,0,None):liver_coords[x,1]+crop_size[1]/2]
                to_app_seg = seg[0:1,0:1,np.clip(liver_coords[x,0]-crop_size[0]/2,0,None):liver_coords[x,0]+crop_size[0]/2,np.clip(liver_coords[x,1]-crop_size[1]/2,0,None):liver_coords[x,1]+crop_size[1]/2]
                crop_collect_img.append(to_app_vol)
                crop_collect_seg.append(to_app_seg)

        else:
            crop_pick=[np.random.randint(crop_size[0]/2,img.shape[2]-crop_size[0]/2),
            np.random.randint(crop_size[1]/2,img.shape[3]-crop_size[1]/2)]
            crop_collect_img.append(img[0:1,0:1,crop_pick[0]:crop_pick[0]+crop_size[0],crop_pick[1]:crop_pick[1]+crop_size[1]])
            crop_collect_seg.append(seg[0:1,0:1,crop_pick[0]:crop_pick[0]+crop_size[0],crop_pick[1]:crop_pick[1]+crop_size[1]])
    return crop_collect_img,crop_collect_seg
    del liver_coords,crop_collect_img,crop_collect_seg,to_app_seg,to_app_vol


""""""""""""""""""""""""""""""""""""""""""""""""
""" Compute weighting """
""""""""""""""""""""""""""""""""""""""""""""""""
USE_WEIGHTS=True
#Compute occurence frequencies to but weight on nodule learning
if USE_WEIGHTS:
    """back_label=0.
    nodule_label=0.
    for seg in segmentations:
        segm = np.load(seg)
        nodule_count=np.sum(segm==2)
        back_count=np.sum(segm!=2)
        nodule_label+=nodule_count
        back_label+=back_count
        print "Checked segm.: ",seg

    label_occurences =  np.array([back_label,nodule_label])
    label_weights    =  (label_occurences[[1,0]])**0.25
    label_weights    =  label_weights/np.sum(label_weights)*2.
    label_weights    =  label_weights.astype(np.float32)
    del segm
    p = os.system('echo %s|sudo -S %s' % (sudoPassword, command))"""
    reweight=1.
    #label_weights=np.array([0.30352455,1.69647551]).astype(np.float32)
    label_weights=np.array([1.,1.]).astype(np.float32)
    label_weights = 2.*np.array([label_weights[0]/reweight,reweight*label_weights[1]]/
                             np.sum([label_weights[0]/reweight,reweight*label_weights[1]]))
    label_weights = label_weights.astype(np.float32)

""""""""""""""""""""""""""""""""""""""""""""""""
""" Compile all theano functions """
""""""""""""""""""""""""""""""""""""""""""""""""
#Necessary symbolic variables
print "Start compilation..."
input_var = theano.tensor.tensor4('input_var')
segmented_gt_vector = theano.tensor.ivector()
weight_vector = theano.tensor.vector()

#Define hyperparameters. These could also be symbolic variables
lr = theano.shared(np.float32(0.000003)) #Shared bc. we decrease it while training.
weight_decay = 1e-4
momentum = 0.95

#L2-regularization
l2_reg_correction = lasagne.regularization.regularize_network_params(net_output_layer, lasagne.regularization.l2) * weight_decay

#If dropout is enabled, we have to distinguish between predictions for test and train sets.
prediction_train = lasagne.layers.get_output(net_output_layer, input_var, deterministic=False)
prediction_test = lasagne.layers.get_output(net_output_layer, input_var, deterministic=True)
prediction_image = lasagne.layers.get_output(imageout,input_var,deterministic = True) #For U-Net

#Final Loss function (we assume that input is flattened to 2D):
loss_train = lasagne.objectives.categorical_crossentropy(prediction_train, segmented_gt_vector)
if USE_WEIGHTS:
    loss_train *= weight_vector
loss_train = loss_train.mean()
loss_train += l2_reg_correction #L2 regularization
#----------
loss_test = lasagne.objectives.categorical_crossentropy(prediction_test, segmented_gt_vector)
if USE_WEIGHTS:
    loss_test *= weight_vector
loss_test = loss_test.mean()
loss_test += l2_reg_correction #L2 regularization

#Compute training accuracy
training_acc  = T.mean(T.eq(T.argmax(prediction_train,axis=1), segmented_gt_vector), dtype=theano.config.floatX)
test_acc= T.mean(T.eq(T.argmax(prediction_test,axis=1), segmented_gt_vector), dtype=theano.config.floatX)

#updates
net_params = lasagne.layers.get_all_params(net_output_layer, trainable=True)

beta1=0.9
beta2=0.999
epsilon=1e-8
updates= lasagne.updates.adam(loss_train, net_params, lr, beta1, beta2, epsilon)

#Training functions
if not USE_WEIGHTS:
    train_fn = theano.function([input_var, segmented_gt_vector], [loss_train, training_acc], updates=updates)
    print "1/3 functions compiled."
    val_fn = theano.function([input_var, segmented_gt_vector], [loss_test, test_acc])
    print "2/3 functions compiled."
else:
    train_fn = theano.function([input_var, segmented_gt_vector,weight_vector], [loss_train, training_acc], updates=updates)
    print "1/3 functions compiled."
    val_fn = theano.function([input_var, segmented_gt_vector,weight_vector], [loss_test, test_acc])
    print "2/3 functions compiled."

make_prediction_image= theano.function([input_var], prediction_image)
print "3/3 functions compiled."
print "Compiling done."
print "------------------------------"


""""""""""""""""""""""""""""""""""""""""""""""""
""" Start training """
""""""""""""""""""""""""""""""""""""""""""""""""
print "Starting training the network..."

# Special Training Flags
liver_only   = True #only train on slices with liver
train_crops  = True
augment_factor = 0 #Augment data by factor augment_factor
DORAND = False #Randomly choose dataset:

#Even more special training flags:
number_crops=2
crops_size  =[64,64]
augment_mode=["flip"]
augment_rand=False

# Actual Training Parameters
n_epochs   = 100
batch_size = 1
num_train  = 110
no_batches = num_train/batch_size
check_every= 20

#Validation parameters
do_val=True
vbatch_size= 1
num_vals   = 20
val_picks  = np.linspace(num_train+1,num_train+num_vals,num_vals).astype(np.int32)

#Misc parameters
batches_per_epoch = no_batches*3/4 #if you don't want to run through all batches


if DORAND:
    np.random.seed(111)
    np.random.shuffle(volumes)
    np.random.seed(111)
    np.random.shuffle(segmentations)

start = time.time()
coll_loss=[]
coll_acc =[]
coll_val =[]
best_loss=0
for epoch in xrange(n_epochs):
    print "Training in epoch: ",epoch+1,"/",n_epochs,"..."
    batchnumber = 0
    training_losses=[]
    training_accs  =[]
    start1 = time.time()
    start2 = time.time()
    print "Currently available GPU memory: {}GB.".format(sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]/1024./1024/1024)
    for batch in xrange(no_batches):
        d_batch = lhf.normalize(lhf.fit_dimensions(np.load(volumes[batch])),use_bd=True)
        l_batch = lhf.fit_dimensions(np.load(segmentations[batch]))
        liver_check = lhf.fit_dimensions(np.load(livers[batch]))
        l_batch[l_batch==2]=1
        if liver_only:
            d_batch,l_batch,liver_check=do_liver_only_slices(d_batch,l_batch,liver_check)
        d_batch = d_batch.astype(np.float32)
        l_batch = l_batch.astype(np.int32)

        training_losses_slices=[]
        training_accs_slices=[]
        for i in xrange(d_batch.shape[2]):
            l_batch_slice_flattened = l_batch[:,:,i,:,:].ravel()
            d_batch_slice           = d_batch[:,:,i,:,:]

            if USE_WEIGHTS:

                if train_crops:
                    train_crops_loss = []
                    train_crops_acc  = []
                    d_crop,l_crop = crop_2D(d_batch_slice,l_batch[:,:,i,:,:],liver_check[:,:,i,:,:],seeed=(epoch*700)+batch,no_crops=number_crops,crop_size=crops_size,only_in_liver=True)

                    for crops in xrange(len(d_crop)):
                        train_loss, train_acc = train_fn(d_crop[crops], l_crop[crops].ravel(),label_weights[l_crop[crops].ravel()])
                        train_crops_loss.append(train_loss)
                        train_crops_acc.append(train_acc)
                    training_losses_slices.append(np.mean(train_crops_loss))
                    training_accs_slices.append(np.mean(train_crops_acc))
                else:
                    train_loss, train_acc = train_fn(d_batch_slice, l_batch.ravel(),label_weights[l_batch_slice_flattened])
                    training_losses_slices.append(train_loss)
                    training_accs_slices.append(train_acc)
                if augment_factor:
                    for i in xrange(augment_factor):
                        if train_crops:
                            train_crops_loss = []
                            train_crops_acc  = []
                            d_crop,l_crop = crop_2D(augment_data(d_batch_slice,seeed=(i+1)*batch,mode=augment_mode,rand_choice=augment_rand,to_type=np.float32),
                                                    augment_data(l_batch[:,:,i,:,:],seeed=(i+1)*batch,mode=augment_mode,rand_choice=augment_rand,to_type=np.int32),
                                                    no_crops=number_crops,crop_size=crops_size)
                            for crops in xrange(len(d_crop)):
                                train_loss, train_acc = train_fn(d_crop[crops], l_crop[crops].ravel(),label_weights[l_crop[crops].ravel()])
                                train_crops_loss.append(train_loss)
                                train_crops_acc.append(train_acc)
                            training_losses_slices.append(np.mean(train_crops_loss))
                            training_accs_slices.append(np.mean(train_crops_acc))
                        else:
                            train_loss,train_acc = train_fn(augment_data(d_batch_slice,seeed=(i+1)*batch,mode=augment_mode,rand_choice=augment_rand,to_type=np.float32),
                                                            augment_data(l_batch[:,:,i,:,:],seeed=(i+1)*batch,mode=augment_mode,rand_choice=augment_rand,to_type=np.int32))
                            training_losses_slices.append(train_loss)
                            training_accs_slices.append(train_acc)

            else:
                if train_crops:
                    train_crops_loss = []
                    train_crops_acc  = []
                    d_crop,l_crop = crop_2D(d_batch_slice,l_batch[:,:,i,:,:],no_crops=number_crops,crop_size=crops_size)
                    for crops in xrange(len(d_crop)):
                        train_loss, train_acc = train_fn(d_crop[crops], l_crop[crops].ravel())
                        train_crops_loss.append(train_loss)
                        train_crops_acc.append(train_acc)
                    training_losses_slices.append(np.mean(train_crops_loss))
                    training_accs_slices.append(np.mean(train_crops_acc))
                else:
                    train_loss, train_acc = train_fn(d_batch_slice, l_batch_slice_flattened)
                    training_losses_slices.append(train_loss)
                    training_accs_slices.append(train_acc)

                if augment_factor:

                    for i in xrange(augment_factor):
                        if train_crops:
                            train_crops_loss = []
                            train_crops_acc  = []

                            d_crop,l_crop = crop_2D(augment_data(d_batch_slice,seeed=(i+1)*batch,mode=augment_mode,rand_choice=augment_rand,to_type=np.float32),
                                                    augment_data(l_batch[:,:,i,:,:],seeed=(i+1)*batch,mode=augment_mode,rand_choice=augment_rand,to_type=np.int32),
                                                    no_crops=number_crops,crop_size=crops_size)

                            for crops in xrange(len(d_crop)):
                                train_loss, train_acc = train_fn(d_crop[crops], l_crop[crops].ravel())
                                train_crops_loss.append(train_loss)
                                train_crops_acc.append(train_acc)

                            training_losses_slices.append(np.mean(train_crops_loss))
                            training_accs_slices.append(np.mean(train_crops_acc))
                        else:
                            train_loss,train_acc = train_fn(augment_data(d_batch_slice,seeed=(i+1)*batch,mode=augment_mode,rand_choice=augment_rand,to_type=np.float32),
                                                            augment_data(l_batch[:,:,i,:,:],seeed=(i+1)*batch,mode=augment_mode,rand_choice=augment_rand,to_type=np.int32))
                            training_losses_slices.append(train_loss)
                            training_accs_slices.append(train_acc)



        training_losses.append(np.mean(training_losses_slices))
        training_accs.append(np.mean(training_accs_slices))
        batchnumber += 1
        #if batchnumber==batches_per_epoch:
        #    break
        if batch%(no_batches/(num_train/check_every))==0 and batch!=0:
            end2 = time.time()
            print "Completed training on {}/{} batches.".format(batch,no_batches)
            print "This took {0:.2f}s.".format(end2-start2)
            #print "This Op. used %s GBs of GPU mem."%(str(GPU_mem_avail2-GPU_mem_avail))
            start2 = time.time()

        p = os.system('echo %s|sudo -S %s' % (sudoPassword, command))
    end1 = time.time()
    coll_loss.append(np.mean(training_losses))
    coll_acc.append(np.mean(training_accs))
#    if epoch%20==0 and epoch!=0:
#        lr *= 0.1
    lr *= 0.985
    print "Finished epoch with:\ntrain accuracy: ",np.mean(training_accs),", and train loss: ",np.mean(training_losses),"."
    print "In total: roughly {0:.2f}s.".format(end1-start1)
    if do_val:
        print "Now validating..."
        startv = time.time()
        total_val_acc = 0
        total_val_loss= 0
        total_val_acc_temp = 0
        total_val_loss_temp= 0

        for vbatch in val_picks:
            val_losses_slices=[]
            val_accs_slices=[]
            val_d = lhf.normalize(lhf.fit_dimensions(np.load(volumes[vbatch])),use_bd=True)
            val_l = lhf.fit_dimensions(np.load(segmentations[vbatch]))
            val_l[val_l==2]=1
            liver_check_val = lhf.fit_dimensions(np.load(livers[vbatch]))
            liver_check_val.shape
            if liver_only:
                val_d, val_l,liver_check_val=do_liver_only_slices(val_d,val_l,liver_check_val)
            val_d = val_d.astype(np.float32)
            val_l = val_l.astype(np.int32)
            for i in xrange(val_d.shape[2]):
                val_d_slice=val_d[:,:,i,:,:]
                val_l_slice=val_l[:,:,i,:,:]
                if USE_WEIGHTS:
                    val_d_crop,val_l_crop = crop_2D(val_d_slice,val_l_slice,liver_check_val[:,:,i,:,:],seeed=vbatch,no_crops=number_crops,crop_size=crops_size,only_in_liver=True)
                    temp_val_loss = []
                    temp_val_acc = []
                    for k in xrange(len(val_d_crop)):
                        ll,acc = val_fn(val_d_crop[k],val_l_crop[k].ravel(),label_weights[val_l_crop[k].ravel()])
                        temp_val_acc.append(acc)
                        temp_val_loss.append(ll)
                else:
                    ll,acc = val_fn(val_d_slice,val_l_slice.ravel())
                val_losses_slices.append(np.mean(temp_val_loss))
                val_accs_slices.append(np.mean(temp_val_acc))
            total_val_acc_temp += np.mean(val_accs_slices)
            total_val_loss_temp+= np.mean(val_losses_slices)

        total_val_acc = total_val_acc_temp/len(val_picks)
        total_val_loss = total_val_loss_temp/len(val_picks)

        if total_val_acc>best_loss:
            best_loss=total_val_acc
            best_params=lasagne.layers.get_all_param_values(net_output_layer)

        endv = time.time()
        coll_val.append(total_val_acc)
        current_params=lasagne.layers.get_all_param_values(net_output_layer)
        print "Validation accuracy: ",total_val_acc
        print "This took: {0:.2f}s.".format(endv-startv)
    print "-----------------------------------"
    #p = os.system('echo %s|sudo -S %s' % (sudoPassword, command))

end = time.time()
print "Total network training time: {0:.2f}s!".format(end-start)
print "Training Done."

""""""""""""""""""""""""""""""""""""""""""""""""
""" Save best/latest network parameters """
""""""""""""""""""""""""""""""""""""""""""""""""
with open(General_name+"_parameters_by_val.pkl", 'w') as f:
    cPickle.dump(best_params, f)
with open(General_name+"_parameters_by_end.pkl", 'w') as f:
    cPickle.dump(current_params, f)

""""""""""""""""""""""""""""""""""""""""""""""""
""" Plot results """
""""""""""""""""""""""""""""""""""""""""""""""""

x = np.linspace(1,len(coll_loss),len(coll_loss))
f,ax = plt.subplots(1,1,sharex=False,sharey=False)
ax.plot(x,coll_acc,x,coll_val)
ax.legend(["Training accuracy","Validation accuracy"])
ax.set_xlabel("Epoch")
ax.set_title("Network performance checks")
ax2 = ax.twinx()
ax2.plot(x,coll_loss,"r")
ax2.legend(["Training Loss"])
f.set_size_inches(15,5)
f.tight_layout()
#f.savefig(General_name+"_lr00001_vs_trainacc_valacc.png", bbox_inches='tight')
plt.show()

