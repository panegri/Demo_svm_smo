% This file is part of the implementation on MATLAB of the Platt pseudo 
% code published on "Sequential Minimal Optimization: A Fast Algorithm
% for Training Support Vector Machine" paper.
% 
% Copyright(c) 2017 Pablo Negri
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

function procRecogNumber(parametersFolder,img,kernelFunction,NTRAIN,GRADIENT_DIRECTIONS,KNN)

global POLYNOMIAL_DEGREE GAMMA

% Verify MEX compilation
mexfile = ['getHogData_c.' mexext];
if ~exist(fullfile('SVMCode','HoGFunctions',mexfile),'file')
    run(fullfile('SVMCode','HoGFunctions','make.m'));
end

% Verify the existence of HOG Features descriptors
if ~exist(fullfile('SVMCode','data','index_HoG.mat'),'file')
    run(fullfile('SVMCode','HoGFunctions','script_set_HoG_features.m'));
end

% Verify the existence of Features Matrix from License Plate Dataset
if ~exist(fullfile('SVMCode','data','HOGFeatures.mat'),'file')
    run(fullfile('SVMCode','HoGFunctions','script_getLettersHOGFeatures.m'));
end

C = 0.1;
switch kernelFunction
    case 1 % Linear
        str_kernel = 'linear';
    case 2 % Polynomial 
        str_kernel = 'polynomial';
    case 3 % RBF
        str_kernel = 'rbf';
        C = 1;
    case 4 % Optimal Linear
        str_kernel = 'linear';
end
classifier_name = ['svmSMOmultipliers_' num2str(NTRAIN*10) '.mat'];
% Verify if the classifier was already trained
if ~exist(fullfile('SVMCode','data',str_kernel,classifier_name),'file')
    % Initialize constants
    % For POLYNOMIAL
    POLYNOMIAL_DEGREE   = 2;
    % For RBF
    GAMMA               = 0.01;
    mainScript
end

% Now run both classifications

% SVM Classifcation
% launch scrip
procSVMRecogNumber(parametersFolder,img,kernelFunction,NTRAIN);

% KNN Classifcation
% launch scrip
procKNNRecogNumber(parametersFolder,img,kernelFunction,NTRAIN,KNN)

