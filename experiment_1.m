setenv('LC_ALL','C');
addpath(genpath('~/Repos/Enitor'));

clearAllButBP;
close all;

% Set experimental results relative directory name
resdir = '';
mkdir(resdir);

saveResult  = 0;

%% Load dataset

% ds = YearPredictionMSD(463715,51630);
% ds = pumadyn(4096,4096, 32 , 'n' , 'h');
% ds = Adult(4096,4096,'plusMinusOne');
% ds = CTslices(6000,4096);
ds = deepRLSsynth(200,4096);


%% Sequential residual batch KRLS, gaussian kernel

map = @gaussianKernel;
filter = @tikhonov;
filterParGuesses = logspace(0,-5,6);
% mapParGuesses = logspace(0,-5,11);
% 
% filterParStar = {1e-7}%, 1e-7};
% mapParStar = {0.1}%, 0.10};

% filterParStar = {1e-4}%, 1e-7};
% mapParStar = {0.7}%, 0.10};

% filterParStar = {1e-4, 1e-3};
mapParStar = {0.7, 0.1};

iterations = size(mapParStar,2);
batchRank = 200;

alg = sequentialResidualKrls(map , filter , ...
                        'batchRank' , batchRank ,  ...
                        'iterations' , iterations ,  ...
                        'mapParStar' , mapParStar ,  ...
                        'filterParGuesses', filterParGuesses , ...
                        'verbose' , 0);

expKRLSSeq = experiment(alg , ds , 1 , true , saveResult , '' , resdir , 0);
expKRLSSeq.run();
expKRLSSeq.result

%%
figure
hold on
plot(ds.X(ds.testIdx),expKRLSSeq.result.Y);
plot(ds.X(ds.testIdx),expKRLSSeq.result.Ypred);
scatter(ds.X(ds.trainIdx),ds.Y(ds.trainIdx))
legend('Ground truth','Prediction','Training examples')
hold off

%% Sequential residual batch Nystrom KRLS

% map = @nystromUniform;
% filter = @tikhonov;
% % filterParGuesses = logspace(0,-7,8);
% % mapParGuesses = logspace(0,-7,8);
% filterParGuesses = 1e-8;
% mapParGuesses = 1.8889 ;
% iterations = 5;
% batchRank = 1000;
% 
% alg = sequentialResidualNkrls(map , filter , ...
%                         'batchRank' , batchRank ,  ...
%                         'iterations' , iterations ,  ...
%                         'mapParGuesses' , mapParGuesses ,  ...
%                         'filterParGuesses', filterParGuesses , ...
%                         'verbose' , 0 , ...
%                         'storeFullTrainPerf' , storeFullTrainPerf , ...
%                         'storeFullValPerf' , storeFullValPerf , ...
%                         'storeFullTestPerf' , storeFullTestPerf);
% 
% expNysSeq = experiment(alg , ds , 1 , true , saveResult , '' , resdir , 0);
% expNysSeq.run();
% expNysInc.result

%%




