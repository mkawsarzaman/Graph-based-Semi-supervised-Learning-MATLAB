function [predict, error]=lgc(data,gnd,labeled_ind,graph)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% This code is the implementation of the graph transductive learning algorithm
%%% Dependency [neighborhood, weights] = LLE_Weights(X, K, WeightModel);
%%% Input:
%%%       --- data:           Input Dataset (nXd);
%%%       --- gnd:            Ground truth label;
%%%       --- labeled_ind:    the index of labeled samples;
%%%       --- graph:          Graph built for inference;
%%%       --- graph.W         Weight matrix (symetric and non-negative, diagnal elements are zeros);
%%%       --- graph.L         Graph Laplacian (normalized);
%%%       --- graph.IS        Prediction matrix IS=(I-\alpha L)^-1;
%%% Output:
%%%       --- predict:        predicted label;
%%%       --- error  :        prediction error;




gnd=reshape(gnd,1,length(gnd));
data_num=length(gnd);
class_num=length(unique(gnd(gnd~=0)));
label_class=unique(gnd);
dim_num=size(data,2);
unlabeled_ind=setdiff([1:data_num], labeled_ind);

alpha=0.95;

if isempty(graph.IS)
    W=graph.W;
    D=full(diag(sum(W)));
    D1=D^(-0.5);
    S=D1*W*D1;
    I=eye(length(W),length(W));
    IS=(I-alpha*S)^(-1);
else
    IS=graph.IS;
end
    


Y=zeros(data_num,class_num);
for i=1:length(labeled_ind)
    Y(labeled_ind(i),gnd(labeled_ind(i)))=1;    
end
 

F=IS*Y;
[a b]=max(F');
predict=b;  
%error=sum((gnd~=predict)&(gnd~=0))/sum(gnd~=0);

gnd_unlabel=ones(1,length(gnd));
gnd_unlabel(labeled_ind)=0;
error=sum((gnd~=predict)&(gnd~=0)&gnd_unlabel)/sum((gnd~=0)&gnd_unlabel);

% for i=1:size(F,2)
%     errors(i)=sum(predict(gnd==i)~=i)/sum(gnd==i);
% end
% error=mean(errors);

%fprintf('%f\n',error);


