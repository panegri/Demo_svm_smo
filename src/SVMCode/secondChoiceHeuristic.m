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

function i1 = secondChoiceHeuristic(i2)

global E;

E2 = E(i2);

[sE,idx] = sort(E);
if E2 > 0
    % choose the sample with lowest error to maximize the step size |E1 - E2|
    if idx(1) == i2
        i1 = idx(2);
    else
        i1 = idx(1);
    end
else
    % choose the sample with highest error to maximize the step size |E1 - E2|
    if idx(end) == i2
        i1 = idx(end-1);
    else
        i1 = idx(end);
    end
end