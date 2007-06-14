function []=gmm(dataset_suffix, n_classes, workdir, progdir, progname)

%% Default files (should not be change)
% progdir='"C:/Documents and Settings/J-Chris/My Documents/Visual Studio 2005/Projects/torchit/release/';	
%workdir = 'C:/Documents and Settings/J-Chris/My Documents/dossiers/epfl_2006-2007/classes/pattern_classification_and_machine_learning - 6 credits/project/torchit/data/'; 

exec=[progdir, progname, '" '];
args=[' -dir "', workdir, '"'];
% args=[args, ' wdbc 2 '];
args=[args, ' ',dataset_suffix,' ',n_classes,' '];

opts = [];

%save current directory
old_dir = pwd;

%change directory
cd(workdir);

%% Call the MLP trainer
display([exec,opts,args]);
system([exec,opts,args]);

%% Options that we want to tune
% learn_rate=0.0001;
% w_decay=0;
% nhu_vec=[5,10,15,25,50,100];
% n_train=42;
% n_valid=1379;

% %% Set the directory
% old_dir=pwd;
% dir=['simu_t',num2str(n_train),'v',num2str(n_valid),'_nhu_lr',num2str(learn_rate),'_wd',num2str(w_decay)]; 
% system(['mkdir ',dir]);
% cd(dir);
% 
% %% Loop for on the parameters of nhu
% for i=1:6
%     nhu=nhu_vec(i);
%     suffix=num2str(i,'%02.f');
% 
%     %% Set the options to send    
%     opts=['-nhu ',num2str(nhu),...
%         '  -lr ', num2str(learn_rate),...
%         '  -wd ', num2str(w_decay),...
% 	    ' -iter 10000 -suffix _',suffix,...
%         '  -save model_',suffix,'.dat'];
% 
%     %% Call the MLP trainer
%     display([exec,opts,args]);
%     system([exec,opts,args]);
% 
%     %% Loading the files gave by the processor
%     MSE_train = load(['MSE_train_',suffix]);
%     MSE_valid = load(['MSE_valid_',suffix]);
% 
%     %% Building the figure
%     figure(i);
%     epoque=[1:length(MSE_train)];
%     plot(epoque,MSE_train,'b',epoque,MSE_valid,'r'); 
%     legend(['train (',num2str(n_train),' samples)'],['valid (',num2str(n_valid),' samples)']);
%     xlabel('epoques');
%     ylabel('Mean Square Error');
%     title(['BackProp(\alpha)    \alpha=\{nhu = ',num2str(nhu),',lr = ',num2str(learn_rate),',wd = ',num2str(w_decay),'\}']);
%     saveas(gcf,['nhu_',num2str(nhu),'.png']);
%     pause(4);
% end
% 

cd(old_dir);	