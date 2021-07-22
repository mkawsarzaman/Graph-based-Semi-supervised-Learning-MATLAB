function [W, A, L, sig] = AdaptiveGraph(TrainData, K)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Adaptive Graph
% Written by LiuWei
% TrainData(dXn): input data set
% K: unsupervised k-NN, usually fixed to 6;
% A: the graph Laplacian
% L: the normalized graph Laplacian
% sig: deviation
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[dim, N] = size(TrainData); 


%% construct a global graph
A = sparse(N,N);
h = zeros(1,N);
sig = 0;
for i = 1:N
    %[i]
    M = repmat(TrainData(:,i),1,N);
    M = M-TrainData;
    d = sum(M.^2);
    [dis,index] = sort(d,2);
    clear M;
    clear d;
    %A(i,index(2:N)) = dis(2:N);%
    A(i,index(2:K+1)) = dis(2:K+1);
    A(index(2:K+1)',i) = dis(2:K+1)';
    sig = sig+dis(K+1)^0.5; 
    h(i) = dis(K+1);
end
clear TrainData;
clear dis;
clear index;

sig = sig/N;
for i = 1:N
    index = find(A(i,:) > 0);
    %[length(index)]
%     var = h(index);
%     temp = find(var < h(i));
%     var(temp) = h(i);
    A(i,index) = exp(-A(i,index)/(1*sig^2)); %exp(-A(i,index)./var);% %%%% %
%     clear temp;
%     clear var;
%     clear index;
end
clear h;

W=A;
D = sum(A,2);
L = sparse(N,N);
L = diag(D.^-0.5)*A*diag(D.^-0.5);
% for i = 1:N
%     L(i,:) = A(i,:)*(D(i)^-0.5);
% end
% for i = 1:N
%     L(:,i) = L(:,i)*(D(i)^-0.5);
% end
L = speye(N)-L;
L = (L+L')/2;

A = diag(D)-A;
A = (A+A')/2;
clear D;

