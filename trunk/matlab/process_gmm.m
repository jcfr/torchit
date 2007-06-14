function [] = process_gmm( )
%process_gmm Optimize the GMM classifier
%  Detailed explanation goes here

close all;
clear all;

dataset_suffix = 'wdbc';

testing_suffix = '_test';
validation_suffix = '_validation';
training_suffix = '_training';

n_classes = 2;
workdir = 'C:/Documents and Settings/J-Chris/My Documents/dossiers/epfl_2006-2007/classes/pattern_classification_and_machine_learning - 6 credits/project/torchit/data/'; 
progdir = '"C:/Documents and Settings/J-Chris/My Documents/Visual Studio 2005/Projects/torchit/release/';
progname = 'torchit';

gmm(dataset_suffix, int2str(n_classes), workdir, progdir, progname);
