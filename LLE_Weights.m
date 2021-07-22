function [neighborhood, weights] = LLE_Weights(X, K, Metric)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Learning the convex reconstruction weights as in LLE
%  Inputs:
%     X: data as D x N matrix (D = dimensionality, N = #points)
%     K : number of neighbors
%  Outputs:
%     weights: Reconstruction weights learned
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Note: The QP solver can be downloaded from Prof. Yinyu Ye's homepage
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Original version by F. Wang 
%F. Wang, C. Zhang. Label Propagation Through Linear Neighborhoods. Proc. ICML, 2007, pp985-992. CMU,Pittsburgh,Pennsylvania,USA 
% Revised by J. Wang (jwang@ee.columbia.edu) 
% for studying the graph construction approach for SSL based graph methods.
% Dec. 2008

[D,N] = size(X);
%fprintf(1,'LLE running on %d points in %d dimensions\n',N,D);


% STEP1: COMPUTE PAIRWISE DISTANCES & FIND NEIGHBORS 
%fprintf(1,'-->Finding %d nearest neighbours.\n',K);

%if (manner=='text')
if strcmp(Metric,'Cosine');
    for i=1:N
        X(:,i)=X(:,i)/norm(X(:,i));
    end
    distance = ones(N,N)-X'*X;
elseif strcmp(Metric,'Euclidean');
    X2 = sum(X.^2,1);
    distance = repmat(X2,N,1)+repmat(X2',1,N)-2*X'*X;
end

[sorted,index] = sort(distance);
neighborhood = index(2:(1+K),:);

% STEP2: SOLVE FOR RECONSTRUCTION WEIGHTS
%fprintf(1,'-->Solving for reconstruction weights.\n');

if(K>D) 
  %fprintf(1,'   [note: K>D; regularization will be used]\n'); 
  tol=1e-3; % regularlizer in case constrained fits are ill conditioned
else
  tol=0;
end

W = zeros(K,N);
for ii=1:N
   z = X(:,neighborhood(:,ii))-repmat(X(:,ii),1,K); % shift ith pt to origin
   Q = z'*z;                                        % local covariance
   Q = Q + eye(K,K)*tol*trace(Q);                   % regularlization (K>D)
   A = ones(1,K);
   c = zeros(K,1);
   b = 1;
   spsolqp;
   W(:,ii) = x;                           
end;
weights = W;