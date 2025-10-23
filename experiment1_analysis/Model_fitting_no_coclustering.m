%% Read file, do settings for EyeHMM

addpath(genpath('src'))

OPT = struct;
OPT.fixationfile = 'Experiment1_Fix_noEvent.xlsx';
% OPT.imgdir = 'images/';
OPT.imgsize = [1280;720];
OPT.DEBUGMODE = 0;
OPT.S = 1:3;
OPT.imginfo = 'stimuliid_boundaries';
OPT.IMAGE_EXT = 'jpg';
emhmm_toggle_color(1);

[data, SubjNames, TrialNames] = read_xls_fixations(OPT.fixationfile);

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

% Clustering parameters
hemopt.tau = get_median_length(data);
hemopt.seed = 1000;  
hemopt.trials = 300;  
hemopt.initmode = 'gmmNew'; 

disp('====================== Settings Complete ======================')
%% HMM FITTING

[hmms, Ls] = vbhmm_learn_batch(data, OPT.S, vbopt);


disp('====================== HMM Complete ======================')


[group_hmms2] = vhem_cluster(hmms,2, [], hemopt);

disp('====================== HEM cluster Complete ======================')


%% visual

% plot the groups
vhem_plot(group_hmms2, faceimg);

% plot the groups and cluster members
vhem_plot_clusters(group_hmms2, hmms, faceimg);

% plot fixations for each group
vhem_plot_fixations(data, group_hmms2, faceimg);

% plot fixations for each group w/ transition matrix
vhem_plot_fixations(data, group_hmms2, faceimg, 'c');


%% Statistical test %%%%%%%%%%%%%%%%%%%%%%%%%%%
% collect data for group 1 and group 2
data1 = data(group_hmms2.groups{1});
data2 = data(group_hmms2.groups{2});

fprintf('\n*** t-test ***\n');

% run t-test for hmm1 
[p, info, lld] = stats_ttest(group_hmms2.hmms{1}, group_hmms2.hmms{2}, data1);
fprintf('- test group hmm1 different from group hmm2: t(%d)=%0.4g; p=%0.4f; d=%0.3g\n', info.df, info.tstat, p, info.cohen_d);

% run t-test for hmm2
[p, info, lld] = stats_ttest(group_hmms2.hmms{2}, group_hmms2.hmms{1}, data2);
fprintf('- test group hmm2 different from group hmm1: t(%d)=%0.4g; p=%0.4f; d=%0.3g\n', info.df, info.tstat, p, info.cohen_d);


%% get mean log-likelihoods (e.g., to use with a correlation test) %%%%%%%%%
fprintf('\n*** get mean LL for each subject ***\n');

[mLL1] = stats_meanll(group_hmms2.hmms{1}, data);
[mLL2] = stats_meanll(group_hmms2.hmms{2}, data);

fprintf('- mean LL for each subject under group hmm1:\n');
mLL1
fprintf('- mean LL for each subject under group hmm2:\n');
mLL2

% compute AB scale (e.g., to use with correlation test)
AB = (mLL1 - mLL2) ./ (abs(mLL1) + abs(mLL2));
fprintf('- AB scale values:\n');
AB
