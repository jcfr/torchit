function [] = gmm_plot_nll( workdir, result_sub_dir, iterg, n_classes, n_samples, n_gaussians, prefix, suffixes )
%plot_gmm_result Summary of this function goes here
%  Detailed explanation goes here

% scrsz = get(0,'ScreenSize');
% figurepos = [1 scrsz(4)/2 376 266]; 


% create filenames
for i=1:n_classes, 
    training_measures(i) = cellstr(['gmm', prefix, '_', char(suffixes(1)), '_measure_', int2str(i-1)]); 
    load( [workdir, result_sub_dir, char(training_measures(i))] );
    validation_measures(i) = cellstr([ 'gmm', prefix, '_', char(suffixes(2)), '_measure_', int2str(i-1)]);
    load( [workdir, result_sub_dir, char(validation_measures(i))] );
    testing_measures(i) = cellstr([ 'gmm', prefix, '_', char(suffixes(3)), '_measure_', int2str(i-1)]);
    load( [workdir, result_sub_dir, char(testing_measures(i))] );
end; 

%find ymax
for i=1:n_classes, 
%     train = eval( char( training_measures(i) ) ); 
%     val =  eval( char( validation_measures(i) ) ); 
%     test =  eval( char( testing_measures(i) ) ); 
    ymin(i) = min([ eval( char( training_measures(i) ) );  eval( char( validation_measures(i) ) );  eval( char( testing_measures(i) ) ) ]);
    if (ymin(i)<0)
        ymin(i) = ymin(i) + ymin(i) * 0.15;
    else
        ymin(i) = ymin(i) - ymin(i) * 0.15;
    end;
    ymax(i) = max([ eval( char( training_measures(i) ) );  eval( char( validation_measures(i) ) );  eval( char( testing_measures(i) ) ) ]);
    if (ymax(i)<0)
        ymax(i) = ymax(i) - ymax(i) * 0.15;
    else
        ymax(i) = ymax(i) + ymax(i) * 0.15;
    end;
end;

figure(1);
% figure('Position',figurepos);
clf reset; 
for i=1:n_classes, 
    subplot(n_classes,1,i);
    hold on; 
    trainm = eval( char( training_measures(i) ) );
    valm = eval( char( validation_measures(i) ) );
    testm = eval( char( testing_measures(i) ) );
    plot( trainm, 'b');
    plot( valm, 'r');
    plot( testm, 'g');
    ylim([ymin(i) ymax(i)]);
    legend(['training (',int2str(n_samples(1,i)),'samples)'], ['validation (',int2str(n_samples(2,i)),'samples)'], ['testing (',int2str(n_samples(3,i)),'samples)'] ,'Location', 'Best');
    title(['Classe ', int2str(i-1), ' (M=', int2str(n_gaussians),')']); 
    xlabel('EM iterations');
    ylabel('negative log-liklihood');
end;
filename = [workdir, result_sub_dir, 'chart', prefix, '_NLL_all.jpg']; 
saveas(gcf,filename);

figure(2);
% figure('Position',figurepos);
clf reset; 
for i=1:n_classes, 
    subplot(n_classes,1,i);
    hold on; 
    trainm = eval( char( training_measures(i) ) ); 
    valm = eval( char( validation_measures(i) ) );
    plot( trainm, 'b');
    plot( valm, 'r');
    ylim([ymin(i) ymax(i)]);
    legend(['training (',int2str(n_samples(1,i)),'samples)'], ['validation (',int2str(n_samples(2,i)),'samples)'] ,'Location', 'Best');
    title(['Classe ', int2str(i-1), ' (M=', int2str(n_gaussians),')']); 
    xlabel('EM iterations');
    ylabel('negative log-liklihood');
end;
filename = [workdir, result_sub_dir, 'chart', prefix, '_NLL_train-valid.eps']; 
saveas(gcf,filename);

for i=1:n_classes, 
    figure(2 + 10 + i);
    % figure('Position',figurepos);
    clf reset;
    hold on; 
    trainm = eval( char( training_measures(i) ) ); 
    valm = eval( char( validation_measures(i) ) );
    plot( trainm, 'b');
    plot( valm, 'r');
    ylim([ymin(i) ymax(i)]);
    legend(['training (',int2str(n_samples(1,i)),'samples)'], ['validation (',int2str(n_samples(2,i)),'samples)'] ,'Location', 'Best');
    title(['Classe ', int2str(i-1), ' (M=', int2str(n_gaussians),')']); 
    xlabel('EM iterations');
    ylabel('negative log-liklihood');
    filename = [workdir, result_sub_dir, 'chart', prefix, '_NLL_c',int2str(i),'_train-valid.eps']; 
    saveas(gcf,filename);
end;

figure(3);
% figure('Position',figurepos);
clf reset; 
for i=1:n_classes, 
    subplot(n_classes,1,i);
    hold on; 
    testm = eval( char( testing_measures(i) ) ); 
    plot( testm , 'g');
    ylim([ymin(i) ymax(i)]);
    legend(['testing (', int2str(n_samples(3,i)), 'samples)'],'Location', 'Best');
    title(['Classe ', int2str(i-1), ' (M=', int2str(n_gaussians),')']); 
    xlabel('EM iterations');
    ylabel('negative log-liklihood');
end;
filename = [workdir, result_sub_dir, 'chart', prefix, '_NLL_testing.jpg']; 
saveas(gcf,filename);
