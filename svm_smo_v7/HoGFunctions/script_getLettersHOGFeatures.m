% This file is part of the implementation on MATLAB of the Platt pseudo 
% code published on "Sequential Minimal Optimization: A Fast Algorithm
% for Training Support Vector Machine" paper.
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

global D;

folder = fullfile('..','Datasets','BaseOCR_MultiStyle');

dchars = {
    '0';
    '1';
    '2';
    '3';
    '4';
    '5';
    '6';
    '7';
    '8';
    '9'}';


Base = [];
labels = [];
Raw = [];
name = {};

DY = 16;
DX = 12;

load(fullfile('..','data','index_HoG.mat'));

for d=1:length(dchars)

    % get file list
    files = dir(fullfile(folder,dchars{d},'*.bmp'));

    tmp_Base = [];
    tmp_Labels = [];
    tmp_Raw = [];
    tmp_name = {};
    for f=1:length(files)
        filename = fullfile(folder,dchars{d},files(f).name);
        disp(filename);
        Ip = double(imread(filename));
    
        Pat = getFeatureVector(Ip,D,index_HoG);        
        tmp_Base = [tmp_Base Pat(:)];       
        tmp_Labels = [tmp_Labels d-1];
        tmp_name{end+1} = files(f).name;
        
        % resize to patch DX-DY for kNN test
        kI = imresize(Ip,[DY DX],'bilinear');
        tmp_Raw = [tmp_Raw kI(:)/max(kI(:))];
    end
    % verify if there are duplicated samples
    nexamples = length(tmp_Labels);
    for f=1:nexamples
        vector = tmp_Base(:,f);
        error_sample = sum([Base tmp_Base] - repmat(vector,1,nexamples+size(Base,2)));
        x = find(error_sample == 0);
        if length(x) == 1
            vector = tmp_Raw(:,f);
            error_sample = sum([Raw tmp_Raw] - repmat(vector,1,nexamples+size(Raw,2)));
            x = find(error_sample == 0);
            if length(x) == 1                
                Base = [Base tmp_Base(:,f)];
                labels = [labels tmp_Labels(f)];
                Raw = [Raw tmp_Raw(:,f)];
                name{end+1} = tmp_name{f};
            else
                fprintf('(%s) Duplicated sample: %s - %d\n',...
                    dchars{d},files(f).name,length(x));
            end
        else
            fprintf('(%s) Duplicated sample: %s - %d\n',...
                dchars{d},files(f).name,length(x));
        end
    end
end

save(fullfile('..','data','HOGFeatures.mat'),'Base','labels','Raw','name');
