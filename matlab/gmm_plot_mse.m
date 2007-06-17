function [] = gmm_plot_mse( workdir, result_sub_dir, iterg, n_classes, n_samples, prefix, suffixes )
%plot_gmm_result Summary of this function goes here
%  Detailed explanation goes here

% create filenames
for i=1:n_classes, 
    training_measures_mse(i) = cellstr(['gmm', prefix, '_', char(suffixes(1)), '_measure_mse_', int2str(i-1)]); 
    load( [workdir, result_sub_dir, char(training_measures_mse(i))] );
    validation_measures_mse(i) = cellstr([ 'gmm', prefix, '_', char(suffixes(2)), '_measure_mse_', int2str(i-1)]);
    load( [workdir, result_sub_dir, char(validation_measures_mse(i))] );
    testing_measures_mse(i) = cellstr([ 'gmm', prefix, '_', char(suffixes(3)), '_measure_mse_', int2str(i-1)]);
    load( [workdir, result_sub_dir, char(testing_measures_mse(i))] );
end; 

%find ymax
for i=1:n_classes, 
%     train = eval( char( training_measures(i) ) ); 
%     val =  eval( char( validation_measures(i) ) ); 
%     test =  eval( char( testing_measures(i) ) ); 
    ymin(i) = min([ eval( char( training_measures_mse(i) ) );  eval( char( validation_measures_mse(i) ) );  eval( char( testing_measures_mse(i) ) ) ]);
    if (ymin(i)<0)
        ymin(i) = ymin(i) + ymin(i) * 0.15;
    else
        ymin(i) = ymin(i) - ymin(i) * 0.15;
    end;
    ymax(i) = max([ eval( char( training_measures_mse(i) ) );  eval( char( validation_measures_mse(i) ) );  eval( char( testing_measures_mse(i) ) ) ]);
    if (ymax(i)<0)
        ymax(i) = ymax(i) - ymax(i) * 0.15;
    else
        ymax(i) = ymax(i) + ymax(i) * 0.15;
    end;
end;

figure(4);
clf reset; 
for i=1:n_classes, 
    subplot(n_classes,1,i);
    hold on; 
    trainm = eval( char( training_measures_mse(i) ) );
    valm = eval( char( validation_measures_mse(i) ) );
    testm = eval( char( testing_measures_mse(i) ) );
    plot( trainm, 'b');
    plot( valm, 'r');
    plot( testm, 'g');
    ylim([ymin(i) ymax(i)]);
    legend(['training (',int2str(n_samples(1,i)),'samples)'], ['validation (',int2str(n_samples(2,i)),'samples)'], ['testing (',int2str(n_samples(3,i)),'samples)'] ,'Location', 'Best');
    title(['Classe ', int2str(i-1)]); 
    xlabel('EM iterations');
    ylabel('MSE');
end;
filename = [workdir, result_sub_dir, 'chart', prefix, '_MSE_all.png']; 
saveas(gcf,filename);

figure(5);
clf reset; 
for i=1:n_classes, 
    subplot(n_classes,1,i);
    hold on; 
    trainm = eval( char( training_measures_mse(i) ) ); 
    valm = eval( char( validation_measures_mse(i) ) );
    plot( trainm, 'b');
    plot( valm, 'r');
    ylim([ymin(i) ymax(i)]);
    legend(['training (',int2str(n_samples(1,i)),'samples)'], ['validation (',int2str(n_samples(2,i)),'samples)'] ,'Location', 'Best');
    title(['Classe ', int2str(i-1)]); 
    xlabel('EM iterations');
    ylabel('MSE');
end;
filename = [workdir, result_sub_dir, 'chart', prefix, '_MSE_train-valid.png']; 
saveas(gcf,filename);

figure(6);
clf reset; 
for i=1:n_classes, 
    subplot(n_classes,1,i);
    hold on; 
    testm = eval( char( testing_measures_mse(i) ) ); 
    plot( testm , 'g');
    ylim([ymin(i) ymax(i)]);
    legend(['testing (', int2str(n_samples(3,i)), 'samples)'],'Location', 'Best');
    title(['Classe ', int2str(i-1)]); 
    xlabel('EM iterations');
    ylabel('MSE');
end;
filename = [workdir, result_sub_dir, 'chart', prefix, '_MSE_testing.png']; 
saveas(gcf,filename);
