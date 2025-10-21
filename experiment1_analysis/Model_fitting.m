%% Read file, do settings for EyeHMM

addpath(genpath('src'))

OPT = struct;
OPT.fixationfile = 'Experiment1_Fix.xlsx';
% OPT.imgdir = 'images/';
OPT.imgsize = [1024; 768];
OPT.DEBUGMODE = 0;
OPT.S = 1:5;
% OPT.imginfo = 'stimuliid_boundaries';
% OPT.IMAGE_EXT = 'jpg';

[alldata, SubjNames, TrialNames, StimuliNames] = read_xls_fixations2(OPT.fixationfile);

tmp = unique(cat(1,StimuliNames{:}));
StimuliNamesImages = cell(length(tmp),2);
StimuliNamesImages(:,1) = tmp;


% for j=1:length(tmp)
%     StimuliNamesImages{j,2} = imread([OPT.imgdir tmp{j}]);
% end

% NOTE: the order of Stimuli will be alphabetized
[alldataC, TrialNamesC, StimuliNamesC] = sort_coclustering(alldata, TrialNames, StimuliNames);

% remove to avoid confusion
clear alldata TrialNames StimuliNames
close all
[Nstimuli, Nsubjects] = size(alldataC);

% map from StimuliNamesC to entry in StimuliNamesImages
mapC = zeros(1,Nstimuli);
for j=1:Nstimuli
    mapC(j) = find(strcmp(StimuliNamesC{j}, StimuliNamesImages(:,1)));
end
  
% reorder to match StimuliNamesC
StimuliNamesImagesC = StimuliNamesImages(mapC,:);
clear StimuliNamesImages

% vbhmm parameters
vbopt.alpha0 = 0.1;
vbopt.mu0    = OPT.imgsize/2;
vbopt.W0     = 0.005;
vbopt.beta0  = 1;
vbopt.v0     = 5;
vbopt.epsilon0 = 0.1;
vbopt.showplot = 0; 
vbopt.bgimage  = '';
vbopt.learn_hyps = 1;
vbopt.seed = 1000; % random seed
vbopt.numtrials = 300;
vbopt.verbose = 1;

emhmm_toggle_color(1);

% Co-clustering parameters
hemopt.tau = round(get_median_length(alldataC(:))); 
hemopt.seed = 1000;  
hemopt.trials = 300;  
hemopt.initmode = 'gmmNew'; 

disp('====================== Settings Complete ======================')
%% HMM FITTING

for j = 1:Nstimuli
    fprintf('********************** Stimuli %d *****************************\n', j);
   [tmp] = vbhmm_learn_batch(alldataC(j,:), OPT.S, vbopt);
    hmm_subjects(j,:) = tmp;
end

disp('====================== HMM Complete ======================')

%% To be add for re-organization for co-clustering if with-in design


%% Group level (specify to 2, using vhem)

[cogroup_hmms, trials] = vhem_cocluster(hmm_subjects, 2, [], hemopt); % may try bayesian version for data driven selection

disp('====================== Co-clustering Complete ======================')

save("HMM_output/Trained_Model.mat", 'cogroup_hmms', 'new_data','StimuliNamesC');
