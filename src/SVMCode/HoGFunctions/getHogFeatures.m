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

function hog_data = getHogFeatures(m,o,N,index_HoG)
x = 1;y = 1;
[dy,dx] = size(m);

rect = [x y dx dy];
% calculate histogram integral
hh = getHistogramIntegral(m,o,N);
for i=1:N
    hhh(:,:,i) = [zeros(1,dx+1); zeros(dy,1) hh(:,:,i)];
end
% Get HoG data
hog_data = getHogData_c(rect,hhh,N,index_HoG);

