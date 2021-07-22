function W=LLE_Graph(X,K,Metric)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Build a constrained LLE graph 
%%% for Trecvid project 
%%% Dependency [neighborhood, weights] = LLE_Weights(X, K, WeightModel);
%%% Input:
%%%       --- X:         Input Dataset (nXd)
%%%       --- K:         Number of the nearest neighbors
%%%       --- Metric:   'Euclidean' - use Euclidean distance evaluate the similarity between samples;
%%%                     'Cosine'    - use cosine distance to evaluate the similarity between samples.

%%% Output:
%%%       --- W: sparse version of weight matrix
%%% by Jun Wang (jwang@ee.columbia.edu)
%%% Last update Dec 8, 2008

%%% Constrained LLE to build the reconstruction weights
[neighborhood, weights] = LLE_Weights(X', K, Metric);


[N,dim] = size(X);
W=zeros(N,N);
for i=1:N
    for j=1:K
        W(i,neighborhood(j,i))=weights(j,i);
    end
end
W = max(full(W),full(W)');
W=sparse(W);

% %%%for debug
% fig=figure;
% [i,j,v]=find(W>0);
% clf;plot(X(:,1),X(:,2),'O');
% for k=1:length(i); 
%     line([X(i(k),1) X(j(k),1)],[X(i(k),2) X(j(k),2)]); 
% end;