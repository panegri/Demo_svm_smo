% This file is part of the implementation on MATLAB of the Platt pseudo 
% code published on "Sequential Minimal Optimization: A Fast Algorithm
% for Training Support Vector Machine" paper, corresponding to 
% the "A MATLAB SMO Implementation to Train a SVM Classifier: 
% Application to Multi-Style License Plate Numbers Recognition",
% version 1.0 IPOL article
% 
% Copyright(c) 2016 Pablo Negri
% pnegri@uade.edu.ar
% 
% This file may be licensed under the terms of of the
% GNU General Public License Version 2 (the ``GPL'').
% 
% Software distributed under the License is distributed
% on an ``AS IS'' basis, WITHOUT WARRANTY OF ANY KIND, either
% express or implied. See the GPL for the specific language
% governing rights and limitations.
% 
% You should have received a copy of the GPL along with this
% program. If not, go to http://www.gnu.org/licenses/gpl.html
% or write to the Free Software Foundation, Inc.,
% 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.

global LINEAR POLYNOMIAL RBF
global K E Targets Alphas C b;
global D
global RAW_DATA HOG_DATA KNN

% Set contants
LINEAR              = 1;
POLYNOMIAL          = 2;
RBF                 = 3;

RAW_DATA            = 1;
HOG_DATA            = 2;

KNN                 = 5;

% Choose Kernel Function type
KernelFunction = RBF;
% SMO contant parameter
C = 1;
% HoG number of directions
D = 4;

addpath(['HoGFunctions']);
% Prepare data and functions to run the script
prepareDataAndFunctions

load(fullfile('data','HOGFeatures.mat'));
% Creating train data and test data.
% NTRAIN samples randomly selected are used for train and the others for tests.
NTRAIN = 20;
rand('seed',1);
x = [];
y = [];
t = [];
ytest = [];
r = [];
rtest = [];

% Split randomly samples for training and test
for i=0:9
    idx = find(labels == i);
    xp = randperm(length(idx));

    % select equal number of samples from each country
    country_count = zeros(4);
    for j=1:length(xp)        
        save4train = 0;
        if isArg(name{idx(xp(j))}) && country_count(1) < NTRAIN/4
            save4train = 1;
            country_count(1) = country_count(1) + 1;
        elseif isEEUU(name{idx(xp(j))}) && country_count(2) < NTRAIN/4
            save4train = 1;
            country_count(2) = country_count(2) + 1;
        elseif isPak(name{idx(xp(j))}) && country_count(3) < NTRAIN/4
            save4train = 1;
            country_count(3) = country_count(3) + 1;
        elseif country_count(4) < NTRAIN/4
            save4train = 1;
            country_count(4) = country_count(4) + 1;
        end
        if save4train
            x = [x Base(:,idx(xp(j)))];
            y = [y labels(idx(xp(j)))];
            r = [r Raw(:,idx(xp(j)))];
        else
            t = [t Base(:,idx(xp(j)))];
            ytest = [ytest labels(idx(xp(j)))];
            rtest = [rtest Raw(:,idx(xp(j)))];
        end
    end
end
%%%%%%%%%%% kNN TEST Using Features %%%%%%%%
MatConfRaw = applyKNN(r,y,rtest,ytest,RAW_DATA);
for k=1:10
    MatConfRaw(k,:) = 100 * MatConfRaw(k,:) / sum(MatConfRaw(k,:));
end

MatConfHOG = applyKNN(x,y,t,ytest,HOG_DATA);
for k=1:10
    MatConfHOG(k,:) = 100 * MatConfHOG(k,:) / sum(MatConfHOG(k,:));
end

%%%%%%%%%%%% TRAIN %%%%%%%%%%%
disp('Training...');
[nfts,npts] = size(x);
% Polinomial based Kernel computation from features vectors.
for i=1:npts
    x1 = x(:,i);
    for j=1:npts
        x2 = x(:,j);
        K(i,j) = computeKernelFunction(x1,x2,KernelFunction);
    end
end
% Train each class against all the others and save the alpha values in a
% matrix.
TabAlphas = zeros(10,length(y));
TabB = zeros(1,10);

for i=1:10
    fprintf('Training character number %s \n',num2str(i-1));
    fprintf('------------------------------\n');
    % create Targets vector
    Targets = -1*ones(1,length(y));
    idx = find(y == (i-1));
    Targets(idx) = 1;
    
    b = 0;
    E = -Targets; % error table
    Alphas = zeros(npts,1);

    % run the support vector algorithm
    computeSMOMultipliers;
    % save computed variables on Tables
    TabAlphas(i,:) = Alphas';
    TabB(i) = b;
end
%%%%RELIABILITY MEASURES%%%%%
[CR_TH,CD_TH] = computeOCRCogniteThresholds(x,y,TabAlphas,TabB,KernelFunction);
%%%%%%%%%%%% TEST %%%%%%%%%%%
disp('Testing...');
o = zeros(size(ytest));
cr = zeros(size(ytest));
cd = zeros(size(ytest));
for i=1:length(ytest)
    xt = t(:,i);
    for k=1:10
        % Create Targets vector
        Targets = -1*ones(1,length(y));
        idx = find(y == (k-1));
        Targets(idx) = 1;
        
        cum_out = 0;
        for j=1:length(y)
            cum_out = cum_out + computeKernelFunction(x(:,j),xt,KernelFunction) * TabAlphas(k,j) * Targets(j);
        end
        vec_outputs(k) = cum_out - TabB(k);
    end
    [mx,cl_mx] = max(vec_outputs);
    o(i) = cl_mx - 1;
    % CONFIDENCE COMPUTATION
    [sort_outputs,srt_mx] = sort(vec_outputs,'descend');
    %%%%% ciarp confidences %%%%
    m = mean(sort_outputs);
    v = var(sort_outputs);
    pdf_outputs = (sort_outputs - m).^2 / v;    
    % normalize to obtain a pdf
    pdf_outputs = pdf_outputs / sum(pdf_outputs);
    cr(i) = pdf_outputs(1) / (sum(pdf_outputs(2:end)));
    cd(i) = pdf_outputs(1);
end

save(fullfile('data','svmSMOmultipliers.mat'),'x','y','TabB','TabAlphas','CR_TH','CD_TH');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Give results as a Confusion Matrix
MatConf = zeros(10);

for i=1:length(ytest)
    MatConf(ytest(i)+1,o(i)+1) = MatConf(ytest(i)+1,o(i)+1) + 1;
end

for k=1:10
    MatConf(k,:) = 100 * MatConf(k,:) / sum(MatConf(k,:));
end
fprintf('Show results\n\n');
fprintf('Confusion Matrix\n');
fprintf('-------------------------------------------------------------------------------------\n');
fprintf('|   |   0   |   1   |   2   |   3   |   4   |   5   |   6   |   7   |   8   |   9   |\n'); 
fprintf('-------------------------------------------------------------------------------------\n');

for k=1:10
    fprintf('| %d | %5s | %5s | %5s | %5s | %5s | %5s | %5s | %5s | %5s | %5s |\n',...
        k-1,...
        sprintf('%3.1f',MatConf(k,1)),sprintf('%3.1f',MatConf(k,2)),...
        sprintf('%3.1f',MatConf(k,3)),sprintf('%3.1f',MatConf(k,4)),...
        sprintf('%3.1f',MatConf(k,5)),sprintf('%3.1f',MatConf(k,6)),...
        sprintf('%3.1f',MatConf(k,7)),sprintf('%3.1f',MatConf(k,8)),...
        sprintf('%3.1f',MatConf(k,9)),sprintf('%3.1f',MatConf(k,10)));
    fprintf('-------------------------------------------------------------------------------------\n');
end

fprintf('\n\n');
fprintf('Overall performance: %2.2f \n',mean(diag(MatConf)));

fprintf('--------------------------------------------------\n');
fprintf('| Method                          | Accuracy (%%) |\n')
fprintf('--------------------------------------------------\n');
fprintf('| Raw data - Knn classification   | %12s | \n', sprintf('%3.1f',mean(diag(MatConfRaw)) ));
fprintf('| HOG data - Knn classification   | %12s | \n', sprintf('%3.1f',mean(diag(MatConfHOG)) ));
fprintf('| HOG data - SVM classification   | %12s | \n', sprintf('%3.1f',mean(diag(MatConf)) ));
fprintf('--------------------------------------------------\n');

fprintf('-------------------------------------------------------------------------------------\n');

fprintf('Reliability Measures\n');
cr_valid = double(cr / CR_TH);
cd_valid = double(cd / CD_TH);
valid_tot = double((cr_valid .* cd_valid) >= 1) ;

fprintf('Percentage of samples with validated outputs: %2.2f \n',...
    100 * sum(valid_tot)/length(valid_tot));