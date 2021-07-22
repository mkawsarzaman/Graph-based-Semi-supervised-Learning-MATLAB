function W = Reconstruction_Weights(X,neighborhood);
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
%[K,N] = size(neighborhood);
for i=1:N
    K(i)=length(neighborhood{i});
end
K=round(mean(K));
%fprintf(1,'LLE running on %d points in %d dimensions\n',N,D);


% STEP1: COMPUTE PAIRWISE DISTANCES & FIND NEIGHBORS 
%fprintf(1,'-->Finding %d nearest neighbours.\n',K);



% STEP2: SOLVE FOR RECONSTRUCTION WEIGHTS
%fprintf(1,'-->Solving for reconstruction weights.\n');

if(K>D) 
  %fprintf(1,'   [note: K>D; regularization will be used]\n'); 
  tol=1e-3; % regularlizer in case constrained fits are ill conditioned
else
  tol=0;
end


for ii=1:N
   %z = X(:,neighborhood(:,ii))-repmat(X(:,ii),1,K); % shift ith pt to origin
   K=length(neighborhood{ii});
   z = X(:,neighborhood{ii})-repmat(X(:,ii),1,K);
   Q = z'*z;                                        % local covariance
   Q = Q + eye(K,K)*tol*trace(Q);                   % regularlization (K>D)
   A = ones(1,K);
   c = zeros(K,1);
   b = 1;
   spsolqp;
   %W(:,ii) = x;       
   weights{ii}=x;
end;



W=zeros(N,N);
for i=1:N
    nn=neighborhood{i};
    ww=weights{i};
    for j=1:length(nn)
        W(i,nn(j))=ww(j);
    end
end

W=sparse(W);

% weights = W;
% 
% W=zeros(N,N);
% for i=1:N
%     for j=1:K
%         W(i,neighborhood(j,i))=weights(j,i);
%     end
% end
% W=sparse(W);
