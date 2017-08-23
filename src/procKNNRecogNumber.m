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

function procKNNRecogNumber(parametersFolder,img,kernelFunction,NTRAIN,KNN)

fts = load(fullfile('SVMCode','data','index_HoG.mat'));
str_kernel = [];
d = [];
gamma = [];

switch kernelFunction
    case 1 % Linear
        str_kernel = 'linear';
    case 2 % Polynomial 
        str_kernel = 'polynomial';
    case 3 % RBF
        str_kernel = 'rbf';
    case 4 % Optimal Linear
        str_kernel = 'linear';
end
        
classifier_name = ['svmSMOmultipliers_' num2str(NTRAIN*10) '.mat'];
svm = load(fullfile('SVMCode','data',str_kernel,classifier_name));

reliability = 0;
number = -1;

Pat = getFeatureVector(img,svm.GRADIENT_DIRECTIONS,fts.index_HoG);
xt = Pat(:);

% Knn classification
tic    
knn_number = applyKNNonHOG(svm.x,svm.y,xt,KNN);
tElapsed = toc;
save(fullfile(parametersFolder,'knn_tElapsed.dat'),'tElapsed','-ascii');
save(fullfile(parametersFolder,'knn_number.dat'),'knn_number','-ascii');
