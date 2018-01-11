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

function img_int = imageIntegrale(img)
[u,v,d] = size(img);
if d > 1
    img = rgb2gray(img);
end
img = double(img);
s = [zeros(1,v+1) ; zeros(u,1) cumsum(img,2)];
ii = cumsum(s,1);
img_int=ii(2:u+1,2:v+1);