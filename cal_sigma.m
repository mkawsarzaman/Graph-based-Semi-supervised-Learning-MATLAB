function sig=cal_sigma(X,K)
N=size(X,1);

KK=X*X'; 
D=diag(KK)*ones(1,N)+ones(N,1)*diag(KK)'-2*KK; 
D=sqrt(D);
D=sort(D);

sig=mean(D(K+1,:));