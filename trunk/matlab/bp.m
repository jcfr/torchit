close all;
clear all;

%% Default files (should not be change)
torchit='/home/neub/epfl/PCML/torchit/trunk/';	
exec=[torchit,'src/Linux_opt_float/bp --train '];
args=[' -valid ',torchit,'data/valid ', torchit,'data/training 19 7'];

%% Options that we want to tune
learn_rate=0.01;
w_decay=0;
nhu_vec=[5,10,15,25,50,100];

%% Set the directory
old_dir=pwd;
dir=['simu_nhu_lr',num2str(learn_rate),'_wd',num2str(w_decay)]; 
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
    legend('train','valid');
    xlabel('epoques');
    ylabel('Mean Square Error');
    title(['BackProp(\alpha)    \alpha=\{nhu = ',num2str(nhu),',lr = ',num2str(learn_rate),',wd = ',num2str(w_decay),'\}']);
    saveas(gcf,['nhu_',num2str(nhu),'.png']);
    pause(4);
end

cd(old_dir);	
