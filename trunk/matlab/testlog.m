function [  ] = testlog(  )
%testlog Summary of this function goes here
%  Detailed explanation goes here


figure(2); 
proba = 0.00001:0.01:1; 
hold on; 
plot(-log(proba), 'b'); 
plot(proba, 'r');
%plot(1-proba, 'b');
plot(log(proba),'g'); 
%plot(log(1-proba),'y'); 
%legend('proba', '1-proba', 'log(proba)', 'log(1-proba)'); 
legend('-log(proba)', 'proba', 'log(proba)'); 