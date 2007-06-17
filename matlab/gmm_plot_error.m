function [ ] = gmm_plot_error( workdir, result_sub_dir, n_classes, n_samples, n_experiments,  suffixes )
%gmm_plot_error Summary of this function goes here
%  Detailed explanation goes here

% create filenames
for i=1:n_classes, 
    training_error(i) = cellstr(['gmm', '_', char(suffixes(1)), '_error_', int2str(i-1)]); 
    load( [workdir, result_sub_dir, char(training_error(i))] );
    validation_error(i) = cellstr([ 'gmm', '_', char(suffixes(2)), '_error_', int2str(i-1)]);
    load( [workdir, result_sub_dir, char(validation_error(i))] );
    testing_error(i) = cellstr([ 'gmm', '_', char(suffixes(3)), '_error_', int2str(i-1)]);
    load( [workdir, result_sub_dir, char(testing_error(i))] );
end; 


figure(20);
clf reset; 
for i=1:n_classes, 
    subplot(n_classes,1,i);
    hold on; 
    traine = eval( char( training_error(i) ) );
    vale = eval( char( validation_error(i) ) );
    teste = eval( char( testing_error(i) ) ); 
    plot( traine(:,1), traine(:,2), 'b');
    plot( vale(:,1), vale(:,2), 'r');
    plot( teste(:,1), teste(:,2), 'g');
    ylim([0 100]); 
    legend(['training (',int2str(n_samples(1,i)),'samples)'], ['validation (',int2str(n_samples(2,i)),'samples)'], ['testing (',int2str(n_samples(3,i)),'samples)'], 'Location', 'Best');
    title(['Classe ', int2str(i-1)]); 
    xlabel('N gaussians');
    ylabel('Classification Error (%)');
end;
filename = [workdir, result_sub_dir, 'chart_ERROR_all.jpg']; 
saveas(gcf,filename);

figure(21);
clf reset; 
for i=1:n_classes, 
    subplot(n_classes,1,i);
    hold on; 
    traine = eval( char( training_error(i) ) );
    vale = eval( char( validation_error(i) ) );
    plot( traine(:,1), traine(:,2), 'b');
    plot( vale(:,1), vale(:,2), 'r');
    ylim([0 100]); 
    legend(['training (',int2str(n_samples(1,i)),'samples)'], ['validation (',int2str(n_samples(2,i)),'samples)'], 'Location', 'Best');
    title(['Classe ', int2str(i-1)]); 
    xlabel('N gaussians');
    ylabel('Classification Error (%)');
end;
filename = [workdir, result_sub_dir, 'chart_ERROR_train-valid.jpg']; 
saveas(gcf,filename);

figure(22);
clf reset; 
traine = zeros(n_experiments,1);
vale = zeros(n_experiments,1);

for i=1:n_classes, 
    current_train = eval( char( training_error(i) ) ); 
    current_val = eval( char( validation_error(i) ) );
    traine = traine + current_train(:,2);
    vale  = vale + current_val(:,2);
end;
traine = traine./n_classes; 
vale = vale./n_classes; 
hold on; 
plot( traine, 'b');
plot( vale, 'r');
ylim([0 100]); 
legend(['training (',int2str(n_samples(1,i)),'samples)'], ['validation (',int2str(n_samples(2,i)),'samples)'], 'Location', 'Best');
title(['Classe ', int2str(i-1)]); 
xlabel('N gaussians');
ylabel('Classification Error (%)');
filename = [workdir, result_sub_dir, 'chart_ERROR_train-valid_AVG.jpg']; 
saveas(gcf,filename);

figure(23);
clf reset; 
for i=1:n_classes, 
    subplot(n_classes,1,i);
    hold on; 
    teste = eval( char( testing_error(i) ) );
    plot( teste(:,1), teste(:,2), 'g');
    ylim([0 100]); 
    legend(['testing (',int2str(n_samples(3,i)),'samples)'], 'Location', 'Best');
    title(['Classe ', int2str(i-1)]); 
    xlabel('N gaussians');
    ylabel('Classification Error (%)');
end;
filename = [workdir, result_sub_dir, 'chart_ERROR_test.jpg']; 
saveas(gcf,filename);