%%%% Experiments for the JMLR paper (fig.7b)
%%% by Jun Wang(jwang@ee.columbia.edu)

clear;
clc;
close all;
path(path,'ManifoldLearn');
path(path,'GraphConstruct');

load datasets/animal_data.mat;
X=double(data);
label=label';


data_num=length(label);
knn=6;
sig=cal_sigma(X,knn);
Test_num=100;
Labels_num=[60 70 80 90 100];




N=size(X,1);
KK=X*X'; 
DD=diag(KK)*ones(1,N)+ones(N,1)*diag(KK)'-2*KK; 
DD=sqrt(DD);
class_num=length(unique(label));

%%% KNN graph
fprintf('KNN graph constructing ... \n');
connectionoptions.KernelType='Linear';    
connectionoptions.Display=0;
connectionoptions.KB=knn;    
connectionoptions.Model='KNN';
[P, Pnn]=ConnectionModel(X,connectionoptions);
  

gnd=label;
%gnd(label==2)=-1;

weightoptions.Type=2;
weightoptions.KB=knn;
weightoptions.Display=0;
weightoptions.KernelSize=sig;
W=WeightingModel(X,P,Pnn,weightoptions);
W=full(W);
graph=TransductionModel(W);
graph.prior=ones(1,class_num);

fprintf('####################KNN Graph K=%d #################### \n', knn);

for labels_i=1:length(Labels_num)
    labnum=Labels_num(labels_i);
    fprintf('The number of labels: %d \n', labnum);
    for test_j=1:Test_num
        ct=cputime;
        labeled_ind=[];
        for ii=1:class_num
            ind_class=find(label==ii);
            rand('seed',(labels_i-1)*Test_num+(test_j-1)*class_num+ii);
            rand_ind=randperm(length(ind_class));
            %labeled_ind=[labeled_ind ind_class(rand_ind(1:labnum))]; 
            labeled_ind=[labeled_ind ind_class(rand_ind(1))]; 
        end

        unlabeled_ind=setdiff([1:data_num],labeled_ind);
        
        if labnum>class_num
            rand_ind=randperm(length(unlabeled_ind));
            labeled_ind=[labeled_ind unlabeled_ind(rand_ind(1:labnum-class_num))];
            unlabeled_ind=setdiff([1:data_num],labeled_ind);
        end
        
       
%         
        [predict, error]=grf_imbalance(X,label,labeled_ind,W);
        knn_grf_error(labels_i,test_j)=error;

        [predict, error]=lgc(X,label,labeled_ind,graph);
        knn_lgc_error(labels_i,test_j)=error;

        [predict, error]=ggmc(X,label,labeled_ind,graph);
        knn_ggmc_error(labels_i,test_j)=error;


    end
    fprintf('LGC-Mean: %f   HFGF-Mean: %f   GGMC-Mean: %f\n',mean(knn_lgc_error(labels_i,:)),mean(knn_grf_error(labels_i,:)),mean(knn_ggmc_error(labels_i,:)));
end


fig=figure; hold on;
X=Labels_num;
plot(X,mean(knn_lgc_error'),'-bd','LineWidth',6,'MarkerSize',30);
plot(X,mean(knn_grf_error'),'-gh','LineWidth',6,'MarkerSize',30);
plot(X,mean(knn_ggmc_error'),'-ro','LineWidth',6,'MarkerSize',30);
legend('LGC','HFGF','GGMC');
xlabel('The number of labels');
ylabel('Error Rate');
grid on;box on;




