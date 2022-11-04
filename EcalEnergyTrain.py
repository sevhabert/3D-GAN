#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This code trains 3D gan and saves loss history and weights in result and weights directories respectively. GANutils is also required.
from __future__ import print_function
from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
from tensorflow import keras
import argparse
import os
os.environ['LD_LIBRARY_PATH'] = os.getcwd()
from six.moves import range
import sys
import glob
import h5py
import numpy as np
import time
import math
import argparse
#import setGPU #if Caltech
import analysis.utils.GANutils as gan # some common functions for gan


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta, Adam, RMSprop
#from tensorflow.keras.utils.generic_utils import Progbar

#import tensorflow as tf
#config = tf.ConfigProto(log_device_placement=True)
#from tensorflow.python.client import timeline
#run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#run_metadata = tf.RunMetadata()

def DivideFiles(FileSearch="/data/LCD/*/*.h5", nEvents=200000, EventsperFile = 10000, Fractions=[.9,.1],datasetnames=["ECAL","HCAL"],Particles=[],MaxFiles=-1):

    Files =sorted( glob.glob(FileSearch))
    Filesused = int(math.ceil(nEvents/EventsperFile))
    FileCount=0

    Samples={}
    for F in Files:
        FileCount+=1
        basename=os.path.basename(F)
        ParticleName=basename.split("_")[0].replace("Escan","")

        if ParticleName in Particles:
            try:
                Samples[ParticleName].append(F)
            except:
                Samples[ParticleName]=[(F)]

        if MaxFiles>0:
            if FileCount>MaxFiles:
                break
    out=[]
    for j in range(len(Fractions)):
        out.append([])

    SampleI=len(Samples.keys())*[int(0)]

    for i,SampleName in enumerate(Samples):
        Sample=Samples[SampleName][:Filesused]
        NFiles=len(Sample)

        for j,Frac in enumerate(Fractions):
            EndI=int(SampleI[i]+ round(NFiles*Frac))
            out[j]+=Sample[SampleI[i]:EndI]
            SampleI[i]=EndI
    return out

#import setGPU #if Caltech
def safe_mkdir(path):
   #Safe mkdir (i.e., don't create if already exists,and no violation of race conditions)
    from os import makedirs
    from errno import EEXIST
    try:
        makedirs(path)
    except OSError as exception:
        if exception.errno != EEXIST:
            raise exception

def GetData(datafile, xscale =1, yscale = 100, dimensions = 3, keras_dformat="channels_last"):
    #get data for training
    print('Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')

    X=np.array(f.get('ECAL'))

    Y=f.get('target')
    Y=(np.array(Y[:,1]))

    X[X < 1e-6] = 0
    X = np.expand_dims(X, axis=-1)
    X = X.astype(np.float32)
    if dimensions == 2:
        X = np.sum(X, axis=(1))

    X = xscale * X

    Y = Y.astype(np.float32)
    Y = Y/yscale
    if keras_dformat !='channels_last':
       X =np.moveaxis(X, -1, 1)
       ecal = np.sum(X, axis=(2, 3, 4))
    else:
       ecal = np.sum(X, axis=(1, 2, 3))
    print('ecal', ecal[:5])
    return X, Y, ecal

def randomize(a, b, c):
    assert a.shape[0] == b.shape[0]
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    shuffled_c = c[permutation]
    return shuffled_a, shuffled_b, shuffled_c

def genbatches(a,n):
    for i in range(0, len(a), n):
        # Create an index range for l of n items:
        yield a[i:i+n]

def main():

    #Architectures to import
    from EcalEnergyGan import generator, discriminator

    #Values to be set by user
    parser = get_parser()
    params = parser.parse_args()
    nb_epochs = params.nbepochs #Total Epochs
    batch_size = params.batchsize #batch size
    latent_size = params.latentsize #latent vector size
    verbose = params.verbose
    #datapath = '/bigdata/shared/LCD/NewV1/*scan/*.h5' #Data path on Caltech
    datapath = params.datapath#Data path on EOS CERN default
    EventsperFile = params.nbperfile#Events in a file
    nEvents = params.nbEvents#Total events for training
    fitmod = params.mod
    weightdir = params.weightsdir
    xscale = params.xscale
    pklfile = params.pklfile
    print(params)
    gan.safe_mkdir(weightdir)

    # Analysis
    analysis=False # if analysing
    energies =[100, 200, 300, 400] # Bins
    resultfile = 'results/3dgan_analysis.pkl' # analysis result

    # Building discriminator and generator
    d=discriminator()
    g=generator(latent_size)
    print("======================================== discriminator ========================================")
    d.summary()
    print("========================================== generator ==========================================")
    g.summary()
    #exit(1)
    Gan3DTrain(d, g, datapath, EventsperFile, nEvents, weightdir, pklfile, resultfile, mod=fitmod, nb_epochs=nb_epochs, batch_size=batch_size, latent_size =latent_size , gen_weight=2, aux_weight=0.1, ecal_weight=0.1, xscale = xscale, analysis=analysis, energies=energies)

    #print("profiling finished. Save to timeline.json")
    #tl = timeline.Timeline(run_metadata.step_stats)
    #ctf = tl.generate_chrome_trace_format()
    #with open("timeline.json", 'w') as f:
    #    f.write(ctf)

# This functions loads data from a file and also does any pre processing
def GetprocData(datafile, xscale =1, yscale = 100, limit = 1e-6):
    #get data for training
    print ('Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')
    Y=f.get('target')
    X=np.array(f.get('ECAL'))
    Y=(np.array(Y[:,1]))
    X[X < limit] = 0
    X = np.expand_dims(X, axis=-1)
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    X = xscale * X
    Y = Y/yscale
    ecal = np.sum(X, axis=(1, 2, 3))
    return X, Y, ecal


@tf.function
def tf_function_discriminator_train(model, x_train, y_train, loss_fn, loss_weight, optimizer):
    with tf.GradientTape() as tape:
        fake, aux, ecal = model(x_train, training=True)
        fake = tf.squeeze(fake)
        loss0 = loss_fn[0](y_train[0], fake)
        loss1 = loss_fn[1](y_train[1], aux)
        loss2 = loss_fn[2](y_train[2], ecal)
        loss = loss0 * loss_weight[0] + loss1 * loss_weight[1] + loss2 * loss_weight[2]
        #print(loss0, loss1, loss2, loss)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return [loss, loss0, loss1, loss2]

@tf.function
def tf_function_generator_predict(generator, x_train):
    return generator(x_train, training = True)

@tf.function
def tf_function_combined_train(generator, discriminator, x_train, y_train, loss_fn, loss_weight, optimizer):
    with tf.GradientTape() as tape:
        fake_image = generator(x_train, training = True)
        fake, aux, ecal = discriminator(fake_image, training = True)
        fake = tf.squeeze(fake)
        loss0 = loss_fn[0](y_train[0], fake)
        loss1 = loss_fn[1](y_train[1], aux)
        loss2 = loss_fn[2](y_train[2], ecal)
        loss = loss0 * loss_weight[0] + loss1 * loss_weight[1] + loss2 * loss_weight[2]
        #print(loss0, loss1, loss2, loss)
    grads = tape.gradient(loss, generator.trainable_variables)
    optimizer.apply_gradients(zip(grads, generator.trainable_variables))
    return [loss, loss0, loss1, loss2]


def Gan3DTrain(discriminator, generator, datapath, EventsperFile, nEvents, WeightsDir, pklfile, resultfile, mod=0, nb_epochs=30, batch_size=128, latent_size=200, gen_weight=6, aux_weight=0.2, ecal_weight=0.1, lr=0.001, rho=0.9, decay=0.0, g_weights='params_generator_epoch_', d_weights='params_discriminator_epoch_', xscale=1, analysis=False, energies=[]):
    start_init = time.time()
    verbose = False
    particle = 'Ele'
    f = [0.9, 0.1]
    print('[INFO] Building discriminator')
    #discriminator.summary()
    discriminator_optimizer = RMSprop()
    discriminator_loss_fn1 = tf.keras.losses.BinaryCrossentropy()
    discriminator_loss_fn2 = tf.keras.losses.MeanAbsolutePercentageError()
    discriminator_loss_fn3 = tf.keras.losses.MeanAbsolutePercentageError()
    discriminator.compile(
        optimizer=discriminator_optimizer,
        loss=[discriminator_loss_fn1, discriminator_loss_fn2, discriminator_loss_fn3],
        loss_weights=[gen_weight, aux_weight, ecal_weight],
        #options=run_options,
        #run_metadata=run_metadata
    )
    # build the generator
    print('[INFO] Building generator')
    #generator.summary()
    generator.compile(
        optimizer=RMSprop(),
        loss='binary_crossentropy'
        #options=run_options,
        #run_metadata=run_metadata
    )

    # build combined Model
    latent = Input(shape=(latent_size, ), name='combined_z')
    fake_image = generator( latent)
    discriminator.trainable = False
    fake, aux, ecal = discriminator(fake_image)
    combined = Model(
        inputs=[latent], ### modif
        outputs=[fake, aux, ecal], ### modif
        name='combined_model'
    )
    combined_optimizer = RMSprop()
    combined_loss_fn1 = tf.keras.losses.BinaryCrossentropy()
    combined_loss_fn2 = tf.keras.losses.MeanAbsolutePercentageError()
    combined_loss_fn3 = tf.keras.losses.MeanAbsolutePercentageError()
    combined.compile(
        #optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        optimizer=combined_optimizer,
        loss=[combined_loss_fn1, combined_loss_fn2, combined_loss_fn3],
        loss_weights=[gen_weight, aux_weight, ecal_weight]
        #options=run_options,
        #run_metadata=run_metadata
    )
    # Getting Data
    Trainfiles, Testfiles = DivideFiles(datapath, nEvents=nEvents, EventsperFile = EventsperFile, datasetnames=["ECAL"], Particles =[particle])
    #Trainfiles, Testfiles = DivideFiles(datapath, nEvents=nEvents, EventsperFile = EventsperFile, datasetnames=["ECAL"], Particles =[particle], Fractions=[.5,.5])

    print(Trainfiles)
    print(Testfiles)
    print("Train files: {0} \nTest files: {1}".format(Trainfiles, Testfiles))

    #Read test data into a single array
    for index, dtest in enumerate(Testfiles):
       if index == 0:
           X_test, Y_test, ecal_test = GetData(dtest, xscale=xscale)
       else:
           X_temp, Y_temp, ecal_temp = GetData(dtest, xscale=xscale)
           X_test = np.concatenate((X_test, X_temp))
           Y_test = np.concatenate((Y_test, Y_temp))
           ecal_test = np.concatenate((ecal_test, ecal_temp))

    for index, dtrain in enumerate(Trainfiles):
        if index == 0:
            #X_train, Y_train, ecal_train = GetData(dtrain, keras_dformat=keras_dformat, xscale=xscale)
            X_train, Y_train, ecal_train = GetData(dtrain, xscale=xscale)
        else:
            #X_temp, Y_temp, ecal_temp = GetData(dtrain, keras_dformat=keras_dformat, xscale=xscale)
            X_temp, Y_temp, ecal_temp = GetData(dtrain, xscale=xscale)
            X_train = np.concatenate((X_train, X_temp))
            Y_train = np.concatenate((Y_train, Y_temp))
            ecal_train = np.concatenate((ecal_train, ecal_temp))

    nb_test = X_test.shape[0]
    nb_train = X_train.shape[0]# Total events in training files
    total_batches = int(nb_train / batch_size)


    train_history = defaultdict(list)
    test_history = defaultdict(list)
    analysis_history = defaultdict(list)

    init_time = time.time()- start_init
    print('Initialization time is {} seconds'.format(init_time))
    for epoch in range(nb_epochs):
        epoch_start = time.time()
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))

        randomize(X_train, Y_train, ecal_train)


        image_batches = genbatches(X_train, batch_size)
        energy_batches = genbatches(Y_train, batch_size)
        ecal_batches = genbatches(ecal_train, batch_size)

        epoch_gen_loss = []
        epoch_disc_loss = []

        for index in np.arange(total_batches):
            start = time.time()
            image_batch = next(image_batches)
            energy_batch = next(energy_batches)
            ecal_batch = next(ecal_batches)
            noise = np.random.normal(0, 1, (batch_size, latent_size))
            sampled_energies = np.random.uniform(0.1, 5,( batch_size, 1))
            generator_ip = np.multiply(sampled_energies, noise)


            #ecal sum from fit
            ecal_ip = gan.GetEcalFit(sampled_energies, particle,mod, xscale)
            generated_images = tf_function_generator_predict(generator, generator_ip)
            discriminator.trainable = True
            real_batch_loss = tf_function_discriminator_train(discriminator, image_batch, [gan.BitFlip(np.ones(batch_size)), energy_batch, ecal_batch],
                    [discriminator_loss_fn1, discriminator_loss_fn2, discriminator_loss_fn3],
                    [gen_weight, aux_weight, ecal_weight],
                    discriminator_optimizer)
            fake_batch_loss = tf_function_discriminator_train(discriminator, generated_images, [gan.BitFlip(np.zeros(batch_size)), sampled_energies, ecal_ip],
                    [discriminator_loss_fn1, discriminator_loss_fn2, discriminator_loss_fn3],
                    [gen_weight, aux_weight, ecal_weight],
                    discriminator_optimizer)
            epoch_disc_loss.append([
                (a + b) / 2 for a, b in zip(real_batch_loss, fake_batch_loss)
            ])
            discriminator.trainable = False

            trick = np.ones(batch_size)
            gen_losses = []
            for _ in np.arange(2):
                noise = np.random.normal(0, 1, (batch_size, latent_size))
                sampled_energies = np.random.uniform(0.1, 5, ( batch_size,1 ))
                generator_ip = np.multiply(sampled_energies, noise)
                ecal_ip = gan.GetEcalFit(sampled_energies, particle, mod, xscale)
                combined_loss = tf_function_combined_train(generator, discriminator, generator_ip, [trick, sampled_energies.reshape((-1, 1)), ecal_ip],
                    [combined_loss_fn1, combined_loss_fn2, combined_loss_fn3],
                    [gen_weight, aux_weight, ecal_weight],
                    combined_optimizer)
                gen_losses.append(combined_loss)

            epoch_gen_loss.append([
                (a + b) / 2 for a, b in zip(*gen_losses)
            ])
            if (index % 100)==0: # and hvd.rank()==0:
                # progress_bar.update(index)
                print('processed {}/{} batches in {}'.format(index + 1, total_batches, time.time() - start))

        # save weights every epoch
        if True:
           safe_mkdir(WeightsDir)

           print ("saving weights of gen")
           generator.save_weights(WeightsDir + '/{0}{1:03d}.hdf5'.format(g_weights, epoch), overwrite=True)

           print ("saving weights of disc")
           discriminator.save_weights(WeightsDir + '/{0}{1:03d}.hdf5'.format(d_weights, epoch), overwrite=True)

           epoch_time = time.time()-epoch_start
           print("The {} epoch took {} seconds".format(epoch, epoch_time))

           #print('The training took {} seconds.'.format(time.time()-epoch_start))
           print('\nTesting for epoch {}:'.format(epoch + 1))


        test_start=time.time()
        noise = np.random.normal(0.1, 1, (nb_test, latent_size))
        sampled_energies = np.random.uniform(0.1, 5, (nb_test, 1))
        generator_ip = np.multiply(sampled_energies, noise)
        generated_images = generator.predict(generator_ip, verbose=False, batch_size=batch_size)
        ecal_ip = gan.GetEcalFit(sampled_energies, particle, mod, xscale)
        sampled_energies = np.squeeze(sampled_energies, axis=(1,))
        X = np.concatenate((X_test, generated_images))
        y = np.array([1] * nb_test + [0] * nb_test)
        ecal = np.concatenate((ecal_test, ecal_ip))
        aux_y = np.concatenate((Y_test, sampled_energies), axis=0)
        discriminator_test_loss = discriminator.evaluate(
            X, [y, aux_y, ecal], verbose=False, batch_size=batch_size)
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        noise = np.random.normal(0.1, 1, (2 * nb_test, latent_size))
        sampled_energies = np.random.uniform(0.1, 5, (2 * nb_test, 1))
        generator_ip = np.multiply(sampled_energies, noise)
        ecal_ip = gan.GetEcalFit(sampled_energies, particle, mod, xscale)
        trick = np.ones(2 * nb_test)
        generator_test_loss = combined.evaluate(generator_ip,
                            [trick, sampled_energies.reshape((-1, 1)), ecal_ip], verbose=False, batch_size=batch_size)
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)
        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)

        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}| {4:5s}'.format(
            'component', *discriminator.metrics_names))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}| {4:<5.2f}'
        print(ROW_FMT.format('generator (train)',
                             *train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)',
                             *test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                             *train_history['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)',
                             *test_history['discriminator'][-1]))

        print("The Testing for {} epoch took {} seconds. Weights are saved in {}".format(epoch, time.time()-test_start, WeightsDir))
        pickle.dump({'train': train_history, 'test': test_history},
        open(pklfile, 'wb'))
        if analysis:
            var = gan.sortEnergy([X_test, Y_test], ecal_test, energies, ang=0)
            result = gan.OptAnalysisShort(var, generated_images, energies, ang=0)
            print('Analysing............')
            # All of the results correspond to mean relative errors on different quantities
            analysis_history['total'].append(result[0])
            analysis_history['energy'].append(result[1])
            analysis_history['moment'].append(result[2])
            print('Result = ', result)
            pickle.dump({'results': analysis_history}, open(resultfile, 'wb'))

def get_parser():
    parser = argparse.ArgumentParser(description='3D GAN Params' )
    parser.add_argument('--model', '-m', action='store', type=str, default='EcalEnergyGan', help='Model architecture to use.')
    parser.add_argument('--nbepochs', action='store', type=int, default=50, help='Number of epochs to train for.')
    parser.add_argument('--batchsize', action='store', type=int, default=128, help='batch size per update')
    parser.add_argument('--latentsize', action='store', type=int, default=200, help='size of random N(0, 1) latent space to sample')
    parser.add_argument('--datapath', action='store', type=str, default='/eos/project/d/dshep/LCD/V1/*scan/*.h5', help='HDF5 files to train from.')
    parser.add_argument('--nbEvents', action='store', type=int, default=200000, help='Number of Data points to use')
    parser.add_argument('--nbperfile', action='store', type=int, default=10000, help='Number of events in a file.')
    parser.add_argument('--verbose', action='store_true', help='Whether or not to use a progress bar')
    parser.add_argument('--weightsdir', action='store', type=str, default='weights/3dganWeights', help='Directory to store weights.')
    parser.add_argument('--mod', action='store', type=int, default=1, help='How to calculate Ecal sum corressponding to energy.\n [0].. factor 50 \n[1].. Fit from Root')
    parser.add_argument('--xscale', action='store', type=int, default=100, help='Multiplication factor for ecal deposition')
    parser.add_argument('--yscale', action='store', type=int, default=100, help='Division Factor for Primary Energy.')
    parser.add_argument('--pklfile', action='store', type=str, default='results/3dgan_history.pkl', help='File to save losses.')
    return parser

if __name__ == '__main__':
    main()
