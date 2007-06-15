close all;
clear all;

%% Default files (should not be change)
torchit='/home/rat/Master2/PCML/Torch3/torchit/torchit/';
torchit='/home/neub/torchit/'
exec=[torchit,'src/Linux_opt_float/bp --train '];
args=[' -valid ',torchit,'data/francoisbobo_db/wdbc_validation.data ', torchit,'data/wdbc_training.data 30 1'];

%% Options that we want to tune
learn_rate=0.001;
w_decay=0;
nhu_vec=[15,25,35,42,50,75,100];
n_train=252;
n_valid=128;

%% Set the directory
old_dir=pwd;
dir=['simu_t',num2str(n_train),'v',num2str(n_valid),'_nhu_lr',num2str(learn_rate),'_wd',num2str(w_decay)]; 
system(['mkdir ',dir]);
cd(dir);

%% Loop for on the parameters of nhu
for i=1:6
    nhu=nhu_vec(i);
    suffix=num2str(i,'%02.f');

    %% Set the options to send    
    opts=['-nhu ',num2str(nhu),...
        '  -lr ', num2str(learn_rate),...
        '  -wd ', num2str(w_decay),...
	    ' -iter 10000 -suffix _',suffix,...
        '  -save model_',suffix,'.dat'];

    %% Call the MLP trainer
    display([exec,opts,args]);
    system([exec,opts,args]);

    %% Loading the files gave by the processor
    MSE_train = load(['MSE_train_',suffix]);
    MSE_valid = load(['MSE_valid_',suffix]);

    %% Building the figure
    figure(i);
    epoque=[1:length(MSE_train)];
    plot(epoque,MSE_train,'b',epoque,MSE_valid,'r'); 
    legend(['train (',num2str(n_train),' samples)'],['valid (',num2str(n_valid),' samples)']);
    xlabel('epoques');
    ylabel('Mean Square Error');
    title(['BackProp(\alpha)    \alpha=\{nhu = ',num2str(nhu),',lr = ',num2str(learn_rate),',wd = ',num2str(w_decay),'\}']);
    saveas(gcf,['nhu_',num2str(nhu),'.png']);
    pause(4);
end

cd(old_dir);	
