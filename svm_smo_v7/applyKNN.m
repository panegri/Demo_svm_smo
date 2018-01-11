% This file is part of the implementation on MATLAB of the Platt pseudo 
% code published on "Sequential Minimal Optimization: A Fast Algorithm
% for Training Support Vector Machine" pape, corresponding to 
% the "A MATLAB SMO Implementation to Train a SVM Classifier: 
% Application to Multi-Style License Plate Numbers Recognition",
% version 1.0 IPOL articler.
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

function MatConf = applyKNN(x,y,xtest,ytest,type)

global RAW_DATA HOG_DATA KNN

ntest = length(ytest);
nbase = length(y);
% initialize confusion matrix
MatConf = zeros(10);

for i=1:ntest
    % get test sample
    xt = xtest(:,i);
    yt = ytest(i);
    switch type
        case RAW_DATA
            % compute all distances
            dst = sum(sqrt((x-repmat(xt,1,nbase)).^2),1);            
        case HOG_DATA
            dst = getHistogramCorrelationDistance(x,repmat(xt,1,nbase));
    end
    % sort distances
    [s,idxmin] = sort(dst);
    
    % now, compute the knn result
    cumclasses = zeros(1,10);
    vecSort = y(idxmin(1:KNN));
    u = unique(vecSort);
    for j=1:length(u)
        cumclasses(u(j)+1) = length(find(vecSort == u(j)));
    end
    [mxVotes,outLabel] = max(cumclasses);
    
    MatConf(yt+1,outLabel) = MatConf(yt+1,outLabel) + 1;
    
end