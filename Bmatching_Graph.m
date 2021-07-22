function W=Bmatching_Graph(X,b,AdaptiveFlag);

[data_num,dim] = size(X);

% options=[];
% options.KernelType='Linear';
% K = constructKernel(X,X,options);
% D=sqrt(diag(K)*ones(1,data_num)+ones(data_num,1)*diag(K)'-2*K); 
% nD=max(max(D))+0.01-D;
% nD=nD-diag(diag(nD));
% [P,B]=bdmatchwrapper(nD,ones(size(nD))-eye(size(nD)),b,b);


    
options = [];
options.Metric = 'Euclidean';
options.NeighborMode = 'KNN';
options.k = 0;
options.WeightMode = 'HeatKernel';
options.t = .1;
options.Adaptive=AdaptiveFlag;
options.bSelfConnected=0;
D = constructW(X,options);
[P,B]=bdmatchwrapper(D,ones(size(D))-eye(size(D)),b,b);



W = max(full(P),full(P)');

%%%for debug
% fig=figure;
% [i,j,v]=find(W>0);
% clf;plot(X(:,1),X(:,2),'O');
% for k=1:length(i); 
%     line([X(i(k),1) X(j(k),1)],[X(i(k),2) X(j(k),2)]); 
% end;