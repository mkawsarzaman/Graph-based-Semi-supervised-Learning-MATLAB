function [P,B] = bdmatchwrapper(W, mask, lb, ub, varargin);

% [P,B] = bdmatchwrapper(W, mask, lb, ub, arg1, arg2,...);
%
% Attempts to find optimal subgraph within degree contraints
% W is the weight matrix of the graph
% mask is the adjacency matrix for the graph, or the sparsity mask on W
% lb is the lower bound on the degrees for each row
% ub is the upper bound on the degrees for each row
% arg1..N are extra arguments to the bdmatch command
% 
% P will be the adjacency matrix of the optimal subgraph
% B will be the log-beliefs at the end of belief propagation
%
% Note that the solution is guaranteed only the graph is bipartite. 
% Otherwise this works as an approximation. See README
%
% Copyright 2008 Bert Huang
% 

%bdmatchpath = '~/files/bdmatch/';
bdmatchpath='';
%%%%%%%%%%%
% Change bdmatchpath to the location of your bdmatch executable
%%%%%%%%%%


fileID = num2str(round(rand*40000));

cmd = sprintf('%s./bdmatch -w weights_%s.txt -d deg_%s.txt -o out_%s.txt %s',...
    bdmatchpath, fileID, fileID, fileID);

cmd=[cmd, ' -damp .09'];

for i=1:length(varargin)
    cmd = [cmd ' ' varargin{i}];
end

truezeros = (mask+W)==1;
[I,J] = find(truezeros);
truezeros = sparse(I,J,NaN*ones(length(I),1),size(W,1),size(W,2));

[I,J,V] = find(W+truezeros); %using NaN as a placeholder

V(isnan(V)) = 0;

out = [size(W,1) nnz(mask)];

save(sprintf('tmp1_%s.txt',fileID),'out','-ascii');

out = [I J V];

save(sprintf('tmp2_%s.txt',fileID),'out','-ascii');

system(sprintf('cat tmp1_%s.txt tmp2_%s.txt > weights_%s.txt',fileID,fileID,fileID));

out = [lb ub];

save(sprintf('deg_%s.txt',fileID),'out','-ascii');

disp(cmd);

system(cmd);

ij = load(sprintf('out_%s.txt',fileID));

cmd = sprintf('rm -f *%s.txt', fileID);
system(cmd);

P = sparse(1+ij(:,1), 1+ij(:,2), ij(:,3));
B = sparse(1+ij(:,1), 1+ij(:,2), ij(:,4));