% This file is part of the implementation on MATLAB of the Platt pseudo 
% code published on "Sequential Minimal Optimization: A Fast Algorithm
% for Training Support Vector Machine" pape, corresponding to 
% the "A MATLAB SMO Implementation to Train a SVM Classifier: 
% Application to Multi-Style License Plate Numbers Recognition",
% version 1.0 IPOL article.
% 
% Copyright(c) 2018 Pablo Negri
% pnegri@uba.dc.ar
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
global POLYNOMIAL_DEGREE GAMMA

% For POLYNOMIAL
POLYNOMIAL_DEGREE   = 2;
% For RBF
GAMMA               = 0.01;

% Verify MEX compilation
mexfile = ['getHogData_c.' mexext];
if ~exist(fullfile('HoGFunctions',mexfile))
    run(fullfile('HoGFunctions','make.m'));
end
% Verify the existence of data folder
if ~exist('data','dir')
    mkdir('data');
end

% Verify the existence of HOG Features descriptors
if ~exist(fullfile('data','index_HoG.mat'))
    run(fullfile('HoGFunctions','script_set_HoG_features.m'));
end

% Verify the existence of Features Matrix from License Plate Dataset
if ~exist(fullfile('data','HOGFeatures.mat'))
    run(fullfile('HoGFunctions','script_getLettersHOGFeatures.m'));
end

