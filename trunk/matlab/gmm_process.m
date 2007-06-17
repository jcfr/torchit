function [] = gmm_process( )
%process_gmm Optimize the GMM classifier
%  Detailed explanation goes here

close all;
clear all;

dataset_suffix = 'wdbc';

training_suffix   = 'training  ';
validation_suffix = 'validation';
testing_suffix    = 'test      ';

n_classes = 2;
workdir = 'C:/Documents and Settings/J-Chris/My Documents/dossiers/epfl_2006-2007/classes/pattern_classification_and_machine_learning - 6 credits/project/torchit/data/'; 
progdir = '"C:/Documents and Settings/J-Chris/My Documents/Visual Studio 2005/Projects/torchit/release/';
progname = 'torchit';

result_dir_prefix = 'gmm'; 
%result_dir = [result_dir_prefix, '_', int2str(n_gaussians), '/'];
result_dir = [result_dir_prefix, '/'];

global n_gaussians threshold prior iterk iterg accuracy

% samples number
n_samples = [ 94 158; 48 80; 70 119];

%parameters
n_gaussians = 17; % number of Gaussians / 10
threshold  = 0.001; % variance threshold / 0.001
prior = 0.001; % prior on the weights / 0.001
iterk = 35; % max number of iterations of KMeans / 25
iterg = 35; % max number of iterations of GMM / 25
accuracy = 0.00001; % end accuracy / 0.00001
verbose = ''; % ' -verbose ';

  
params_init = [verbose, ' -result_dir ', result_dir_prefix, ' -threshold ', num2str(threshold), ' -prior ', num2str(prior), ' -iterk ', int2str(iterk), ' -iterg ', int2str(iterg), ' -e ', num2str(accuracy)]; 

i = n_gaussians; 
%for i=1:20:200
    n_gaussians = i; 
    
    prefix = ['_', int2str(n_gaussians)];
    
    params = [params_init, ' -n_gaussians ', int2str(n_gaussians), ' -prefix ' , prefix, ' ' ]; 
    
    %run gmm
    gmm(params, dataset_suffix, n_classes, workdir, progdir, progname, cellstr([training_suffix; validation_suffix; testing_suffix]));
    
    %plot Measurer result
    % gmm_plot_nll(workdir, result_dir, iterg, n_classes, n_samples, n_gaussians, prefix, cellstr([training_suffix; validation_suffix; testing_suffix])); 
    % gmm_plot_mse(workdir, result_dir, iterg, n_classes, n_samples, prefix, cellstr([training_suffix; validation_suffix; testing_suffix])); 

%end;


%gmm_plot_error(workdir, result_dir, n_classes, n_samples, cellstr([training_suffix; validation_suffix; testing_suffix])); 


