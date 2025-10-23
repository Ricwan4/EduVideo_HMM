%% Read file, do settings for EyeHMM

addpath(genpath('src'))

OPT = struct;
OPT.fixationfile = 'Experiment1_Fix.xlsx';
% OPT.imgdir = 'images/';
OPT.imgsize = [1280;720];
OPT.DEBUGMODE = 0;
OPT.S = 1:3;
OPT.imginfo = 'stimuliid_boundaries';
OPT.IMAGE_EXT = 'jpg';
emhmm_toggle_color(1);

[alldata, SubjNames, TrialNames, StimuliNames] = read_xls_fixations2(OPT.fixationfile);

tmp = unique(cat(1,StimuliNames{:}));
StimuliNamesImages = cell(length(tmp),2);
StimuliNamesImages(:,1) = tmp;

% pre-process
alldata = preprocess_fixations(alldata, StimuliNames, OPT.imginfo, StimuliNamesImages);

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

save("HMM_output/Trained_Model.mat", 'cogroup_hmms', 'alldataC','StimuliNamesC');

%% get significant

Nsubjects = 27;
Nstimuli = 140;
HEM_K = length(cogroup_hmms{1}.hmms);
cg_hmms = cogroup_hmms;
grps = cogroup_hmms{1}.groups;

for i=1:Nsubjects
  for j=1:Nstimuli
    for k=1:HEM_K
      if ~isempty(alldataC{j,i})
        % compute LL: if more than one sample for this stimuli, then average.
        % the LL for each subject of each situli under each model
        LL(i,j,k) = mean(vbhmm_ll(cg_hmms{j}.hmms{k}, alldataC{j,i}));
      else
        LL(i,j,k) = 0;
      end

    end
  end
end

% (assumes K=2)
% LL of subject under model 1 and 2
LL1 = sum(LL(:,:,1),2);  % sum over stimuli
LL2 = sum(LL(:,:,2),2);

% compute AB scale
% since some stimuli are missing, then normalize
AB = (LL1 - LL2) ./ (abs(LL1) + abs(LL2));

stim_LL1 = [];
stim_LL2 = [];
  
for j=1:Nstimuli
    % get LL for group 1 under models 1 and 2
    stim_LL1(:,j,1) = LL(grps{1}, j, 1);
    stim_LL1(:,j,2) = LL(grps{1}, j, 2);
    
    % get LL for group 2 under models 2 and 1
    stim_LL2(:,j,1) = LL(grps{2}, j, 2);
    stim_LL2(:,j,2) = LL(grps{2}, j, 1);
end

fprintf('=== group difference (overall) ===\n');
  
% average LL over stimulus for group 1 under models 1 and 2
LL1_grp1 = mean(stim_LL1(:,:,1),1); % subjects in group under group 1 model
LL1_grp2 = mean(stim_LL1(:,:,2),1); % subjects in group under group 2 model
  
[h1,p1,ci1,stats1] = ttest(LL1_grp1, LL1_grp2, 'tail', 'both');
deffect = computeCohen_d(LL1_grp1,LL1_grp2, 'paired');
fprintf('model1 vs model2 using grp1 data: p=%0.4g, t(%d)=%0.4g, cohen_d=%0.4g\n', ...
p1, stats1.df, stats1.tstat, deffect);
  
% average LL over stimulus for group 2 under models 1 and 2
LL2_grp1 = mean(stim_LL2(:,:,1),1); % subjects in group under group 1 model
LL2_grp2 = mean(stim_LL2(:,:,2),1);% subjects in group under group 2 model
   
[h2,p2,ci2,stats2] = ttest(LL2_grp1, LL2_grp2, 'tail', 'both');
deffect = computeCohen_d(LL2_grp1,LL2_grp2, 'paired');
fprintf('model1 vs model2 using grp2 data: p=%0.4g, t(%d)=%0.4g, cohen_d=%0.4g\n', ...
    p2, stats2.df, stats2.tstat, deffect);

clear LL1 LL2 AB stim_LL1 stim_LL2 LL1_grp1 LL1_grp2 h1 p1 ci1 stats1 deffect LL2_grp1 LL2_grp2 h2 p2 ci2 stat2 cg_hmms HEM_K

%% behavior

inputFile = 'stimuli_questionnaire_scores.tsv';
T = readtable(inputFile, 'FileType', 'text', 'Delimiter', '\t');

G = table(T.participant_id, T.total_domain_questions, T.domain_score, T.total_memory_questions, T.memory_score, ...
    'VariableNames', {'participant_id','total_domain_questions','domain_score','total_memory_questions','memory_score'});

S = varfun(@sum, G, 'GroupingVariables', 'participant_id', ...
    'InputVariables', {'total_domain_questions','domain_score','total_memory_questions','memory_score'});

% accuracy
n = height(S);
domain_accuracy = nan(n,1);
memory_accuracy = nan(n,1);
for i = 1:n
    tdq = S.sum_total_domain_questions(i);
    tmq = S.sum_total_memory_questions(i);
    ds  = S.sum_domain_score(i);
    ms  = S.sum_memory_score(i);

    domain_accuracy(i) = ds / tdq;
    memory_accuracy(i) = ms / tmq;
end

%% A-B scale
% Initialize variables
Nsubjects = 27; 
Nstimuli = 140;

HEM_K = length(cogroup_hmms{1}.hmms); % Should be 2 for this setup
LL = zeros(Nsubjects, Nstimuli, HEM_K);

% Compute log-likelihood for each subject under each model
for i = 1:Nsubjects
    for j = 1:Nstimuli
        if ~isempty(alldataC{j, i})
            for k = 1:HEM_K
                LL(i, j, k) = mean(vbhmm_ll(cogroup_hmms{j}.hmms{k}, alldataC{j, i}));
            end
        else
            LL(i, j, :) = NaN; % Handle missing data
        end
    end
end

% Compute LL1 and LL2 (sum across stimuli for each subject)
LL1 = sum(LL(:, :, 1), 2, 'omitnan');
LL2 = sum(LL(:, :, 2), 2, 'omitnan');

% Compute A-B scale
AB_scale = (LL1 - LL2) ./ (abs(LL1) + abs(LL2));

clear trials LL1 LL2 k j i HEM_K

%% Visualize 

event = 60;
vhem_plot(cogroup_hmms{event},[],'c',[])
% vhem_plot_fixations(alldataC(event,:), cogroup_hmms{event}, [], 'g', 1);

