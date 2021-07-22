function sig=cal_sigma_neighbor(Pnn,DD,knn)

D=DD.*(Pnn | Pnn'); 
D=sort(D);
B=sum(D~=0);
if sum(B>=knn)~=length(D)
    ttt=1;
end
knn_ind=length(D)-B+knn;
for i=1:length(D)
    A(i)=D(knn_ind(i),i);
end
sig=mean(A);