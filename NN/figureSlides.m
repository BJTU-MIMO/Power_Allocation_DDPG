clear all; close all; clc;
load('dataset_maxmin.mat')
%% Plot the simulation results

hold on; box on;

% Conversion in dB for input data
Input_tr_dB = 10*log10(Output_tr_MMMSE_maxmin_cell_1);

Input_tr_dB_normalized_maxmin = (Input_tr_dB - mean(Input_tr_dB,2))./std(Input_tr_dB,0,2);

%% Plot the simulation results
figure(5);
hold on; box on;
Output_tr_MMMSE_maxprod_cell_1 = Output_tr_MMMSE_maxprod_cell_1(1:5,:,:);
histogram(Input_tr_dB_normalized_maxprod(:),'Normalization','probability')
histogram(Input_tr_dB_normalized_maxmin(:),'Normalization','probability')
xlim([-4 4]);

legend('Max-prod','Max-min','Location','SouthEast');

xlabel('Normalized Power in dBW');