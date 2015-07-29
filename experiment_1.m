setenv('LC_ALL','C');
% addpath(genpath('~/Repos/Enitor'));

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
ds = deepRLSsynth(300,4096);


%% Sequential residual batch KRLS, gaussian kernel

map = @gaussianKernel;
filter = @tikhonov;
filterParGuesses = logspace(2,-10,30);
% mapParGuesses = logspace(0,-2,20);
mapParStar = logspace(1,-2,40);

% 
% filterParStar = {1e-7}%, 1e-7};
% mapParStar = {0.1}%, 0.10};

% filterParStar = {1e-4}%, 1e-7};
% mapParStar = {0.7}%, 0.10};

% filterParStar = {1e-4, 1e-3};
% mapParStar = {0.7, 0.1};

iterations = 40;
batchRank = 300;

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

%%  Norms of the coefficient vactors for different scales

% cNorms = cellfun(@norm, alg.c);
% 
% figure
% % hold on 
% semilogx(mapParStar,cNorms);
% title('Norms of the coefficient vectors for different scales')
% xlabel('\sigma')
% ylabel('||c||')

repetitions = 10;
cNorms = zeros(repetitions, iterations);
bestlambdas = [];

for i = 1:repetitions 
    
    ds = deepRLSsynth(300,4096);

    alg = sequentialResidualKrls(map , filter , ...
                            'batchRank' , batchRank ,  ...
                            'iterations' , iterations ,  ...
                            'mapParStar' , mapParStar ,  ...
                            'filterParGuesses', filterParGuesses , ...
                            'verbose' , 0);

    expKRLSSeq = experiment(alg , ds , 1 , true , saveResult , '' , resdir , 0);
    expKRLSSeq.run();
    expKRLSSeq.result

    
    for j = 1:numel(alg.c)
        cNorms(i,j) = alg.c{j}'*alg.Ktrain{j}*alg.c{j};
    end
    
%     [cNorms ; cellfun(@norm, alg.c)];

    bestlambdas = [ bestlambdas ; alg.filterParStar];
end

figure
% hold on 
errorbar(mapParStar,mean(cNorms), std(cNorms));
 set(gca,'XScale','log');
title('Norms of the coefficient vectors for different scales')
xlabel('\sigma')
ylabel('||f||')

% hold off||c||

figure
% hold on 
errorbar(mapParStar,mean(bestlambdas), std(bestlambdas));
 set(gca,'XScale','log');
title('Norms of the coefficient vectors for different scales')
xlabel('\sigma')
ylabel('\lambda*')


%%
figure
hold on 
% errorbar(mapParStar,mean(cNorms), std(cNorms));
plot(mapParStar,exp(mean(log(cNorms))));
 set(gca,'XScale','log');
title('Norms of the coefficient vectors for different scales')
xlabel('\sigma')
% errorbar(mapParStar,mean(bestlambdas)*1e6, std(bestlambdas)*1e6);
plot(mapParStar,exp(mean(log(bestlambdas))));
 set(gca,'YScale','log');
title('Norms of the coefficient vectors for different scales')
xlabel('\sigma')
legend('||f||','\lambda*')


%%
[Ypred,YpredStesp]  = alg.test(ds.X(ds.testIdx));

figure
%     hold on
somma = 0;
for i = 1: numel(YpredStesp)
    somma = somma + YpredStesp{i}
    plot(somma)
    pause
end
% hold off


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




