clear;

Input_tr = [];

%Prepare MR maxmin
Output_tr_MR_maxmin = [];

indexes = 1;

for kk = 1:length(indexes)
    
    ii = indexes(kk);
    
    % Load data file
    load(['MyDataFile_' mat2str(ii) '.mat'])
    
    % BETAAs in the network
    BETAAN_reshaped = reshape(BETAAN,K*M,[]);
    
    % input for NN
    %     input = [real(input_positions_reshaped); imag(input_positions_reshaped); real(BS_positions_rep); imag(BS_positions_rep)];
    input = BETAAN_reshaped;
    
    % Output for NN with MR and MMMSE for all cells
    output_MR_maxmin_all = reshape(Power_coefficient(:,:,:), K*M,[]);
    %     output_MR_maxmin_all_cells = [output_MR_maxmin_all_cells; sum(output_MR_maxmin_all_cells,1)];
    
%     output_MR_maxmin_cell_1 = reshape(output_MR_maxmin(:,1,:), K,[]);
    %     output_MR_maxmin_cell_1 = [output_MR_maxmin_cell_1; sum(output_MR_maxmin_cell_1,1)];
    
    
    % Concatenate input data for NN
    Input_tr = [Input_tr, input]; %#ok<*AGROW>
    
    % Concatenate output data for NN
    Output_tr_MR_maxmin = [Output_tr_MR_maxmin, output_MR_maxmin_all];
    
%     Output_tr_MR_maxmin_cell_1 = [Output_tr_MR_maxmin_cell_1, output_MR_maxmin_cell_1];
    
    
end


% Conversion in dB for input data
% Input_tr_dB = 10*log10(Input_tr);
Input_tr_dB = Input_tr;

Input_tr_normalized = (Input_tr - mean(Input_tr,2))./std(Input_tr,0,2);

Input_tr_dB_normalized = (Input_tr_dB - mean(Input_tr_dB,2))./std(Input_tr_dB,0,2);


Output_tr_MR_maxmin_normalized = (Output_tr_MR_maxmin - mean(Output_tr_MR_maxmin,2))./std(Output_tr_MR_maxmin,0,2);

Output_tr_MR_maxmin_dB = 10*log(Output_tr_MR_maxmin);

Output_tr_MR_maxmin_dB_normalized = (Output_tr_MR_maxmin_dB - mean(Output_tr_MR_maxmin_dB,2))./std(Output_tr_MR_maxmin_dB,0,2);



figure;
hold on; box on;
histogram(Output_tr_MR_maxmin_dB(:))
% histogram(Output_tr_MR_maxmin_dB_normalized_cell_1(:))


save('dataset_maxmin.mat','Input_tr','Input_tr_dB','Input_tr_normalized','Input_tr_dB_normalized',...
    'Output_tr_MR_maxmin',...
    'Output_tr_MR_maxmin_normalized',...
    'Output_tr_MR_maxmin_dB_normalized');
clear;

load('MyDataFile_0.mat')

% BETAAs in the network
BETAAN_reshaped = reshape(BETAAN,K*M,[]);

% input for NN
Input_tr = BETAAN_reshaped;

% Conversion in dB for input data
Input_tr_dB = Input_tr;

Input_tr_normalized = (Input_tr - mean(Input_tr,2))./std(Input_tr,0,2);

Input_tr_dB_normalized = (Input_tr_dB - mean(Input_tr_dB,2))./std(Input_tr_dB,0,2);

save('testset_maxmin.mat','Input_tr','Input_tr_dB','Input_tr_normalized','Input_tr_dB_normalized');
