function [P, Pnn]=ConnectionModel(data,connectionoptions)



[data_num,dim] = size(data);
options=[];
options.KernelType=connectionoptions.KernelType;
K = constructKernel(data,data,options);

D=sqrt(diag(K)*ones(1,data_num)+ones(data_num,1)*diag(K)'-2*K); 
nD=max(max(D))+0.01-D;
nD=nD-diag(diag(nD));


if strcmp(connectionoptions.Model,'BMATCHING')
    b=connectionoptions.KB;
    [Pnn,B]=bdmatchwrapper(nD,ones(size(nD))-eye(size(nD)),b,b);
    P = (Pnn | Pnn'); 
    %P=Pnn+Pnn';
    P=sparse(P);
end
if strcmp(connectionoptions.Model,'KNN')
    knn=connectionoptions.KB;
    Pnn = zeros(size(nD));
    for i=1:data_num
        q=sort(nD(i,:));
        Pnn(i,:) = (nD(i,:)>=q(end-knn+1));
    end
    P = (Pnn | Pnn'); 
    %P=Pnn+Pnn';
    P=sparse(P);
end

