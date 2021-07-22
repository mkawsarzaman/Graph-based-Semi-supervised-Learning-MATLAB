function W=WeightingModel(data,P,Pnn,weightoptions)

[data_num,dim] = size(data);
kb=weightoptions.KB;
weight_type=weightoptions.Type;

switch weight_type
    case 1 %%%Binary Weighting
        W=P;        
    case 2 %%%Global Heat Kernel Weighting
        options = [];
        options.Metric='Euclidean';
        options.NeighborMode = 'KNN';
        options.k = 0;%%%jwang july 2012
        %options.k=kb;
        options.WeightMode = 'HeatKernel';
        options.t = weightoptions.KernelSize;
        options.Adaptive='NO'; %%%jwang july 2012
        %options.Adaptive='YES';
        options.bSelfConnected=0;
        W = constructW(data,options); 
        W =W.*P;
    case 3 %%%Local Scaling for Heat Kernel Weighting
        options=[];
        options.KernelType='Linear';
        K = constructKernel(data,data,options);
        D=sqrt(diag(K)*ones(1,data_num)+ones(data_num,1)*diag(K)'-2*K);
        nD=Pnn.*D;
%         nDD=full(nD);
%         nDD(Pnn==0)=1000;
%         nDD=sort(nDD);
%         kk=max(nDD(1,:))
%         W=zeros(data_num,data_num);
%         W=exp(-D.^2/(2*kk^2));
%         W=W.*Pnn;
        for i=1:data_num;
            idx=find(Pnn(i,:));
            for j=1:length(idx)
                j_ind=idx(j);
                idy=find(Pnn(j_ind,:));
                kk=mean([D(i,idx) D(j_ind,idy)]);
                W(i,j_ind)=exp(-D(i,j_ind)^2/(2*kk^2));
                %W(i,idx)=exp(-nD(i,idx).^2/(2*kk^2));
            end
        end
        W=max(W,W');
        
%         options = [];
%         options.Metric='Euclidean';
%         options.NeighborMode = 'KNN';
%         options.k = kb;
%         options.t = weightoptions.KernelSize;
%         options.Adaptive='YES';
%         options.bSelfConnected=0;
%         options.WeightMode = 'HeatKernel';
%         W = constructW(data,options); 
%         W =W.*P;
    case 4 %%%Reconstruction Weighting
        for i=1:data_num;
            %neighborhood(:,i)=find(Pnn(i,:));
            neighborhood{i}=find(Pnn(i,:));
        end        
        W = Reconstruction_Weights(data',neighborhood);
        W = max(full(W),full(W)');
end

if weightoptions.Display
    figure;
    [i,j,v]=find(W>0);
    clf;
    plot(data(:,1),data(:,2),'O');
    for k=1:length(i); 
        line([data(i(k),1) data(j(k),1)],[data(i(k),2) data(j(k),2)]); 
    end;
end