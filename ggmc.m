%%% this code implements the Greedy Gradient based Max-Cut algorithm


%%% updating by pick up the most possible samples; This version is robust
%%% to weight matrix;


function [predict, error]=ggmc(data,gnd,labeled_ind,graph)

gnd=reshape(gnd,1,length(gnd));
data_num=length(gnd);
class_num=length(unique(gnd(gnd~=0)));
label_class=unique(gnd);
%%% coefficient for local label fitting
%mu=0.01;   
dim_num=size(data,2);
ori_labeled_ind=labeled_ind;

W=full(graph.W);
d=sum(W);
A=graph.A;

ratio=graph.prior;

%%% iteration counter
iter_num=0;
%%% predicted labels;
predict=zeros(1,length(gnd));
predict(labeled_ind)=gnd(labeled_ind);
%%% unlabeled index
unlabeled_ind=setdiff([1:data_num],labeled_ind);

for i=1:class_num
    class_degree(i)=sum(d(predict==i));
end

%%% Initial given labels
Y=zeros(data_num,class_num);
for i=1:length(labeled_ind)
    ii=labeled_ind(i);
    Y(ii,gnd(ii))=1;     
end
originalY=Y;

%%% Initial node regularizer
V=zeros(data_num,data_num);
for i=1:length(labeled_ind)
    ii=labeled_ind(i);
    V(ii,ii)=ratio(predict(ii))*d(ii)/class_degree(predict(ii));   
    %V(ii,ii)=1;
end

normalizedY=V*Y;
DeltaQ=A*normalizedY; 
clear V Y normalizedY;   


while isempty(unlabeled_ind)~=1     
    DeltaY=-DeltaQ(unlabeled_ind,:);  %%%only look into the unlabeled data points
    [a_ind, b_ind]=find(DeltaY==max(max(DeltaY)));
    
    %%%find multiple elements with minimum connectivity
    if length(a_ind)>1
        a_ind=a_ind(1);
        b_ind=b_ind(1);
    end
    new_label_ind=unlabeled_ind(a_ind);
       

        
    %iter_num=iter_num+1
    predict(unlabeled_ind(a_ind))=b_ind; %%% adding the unlabeled point to a subsect
    %error(iter_num-1)=1-(sum((predict==gnd)&(gnd~=0)))/sum(gnd~=0);
    
    %%%update label and unlabeled sets
    labeled_ind=[labeled_ind,unlabeled_ind(a_ind)];
    unlabeled_ind=setdiff([1:data_num],labeled_ind);
   
    %%%update the connectivity C matrix Eq. 46
    DeltaQ(:,b_ind)=DeltaQ(:,b_ind).*class_degree(b_ind)./(class_degree(b_ind)+d(new_label_ind))+A(:,new_label_ind).*d(new_label_ind)./(class_degree(b_ind)+d(new_label_ind));
    class_degree(b_ind)=class_degree(b_ind)+d(new_label_ind);
    
    
end

%error=sum((gnd~=predict)&(gnd~=0))/sum(gnd~=0);
gnd_unlabel=ones(1,length(gnd));
gnd_unlabel(ori_labeled_ind)=0;
error=sum((gnd~=predict)&(gnd~=0)&gnd_unlabel)/sum((gnd~=0)&gnd_unlabel);