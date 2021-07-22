function gvalue=cal_optimalg(X)

[data_num,fea_num]=size(X);
dis_num=0;
rand_ind=randperm(data_num);
%X=X(rand_ind(1:500),:);
[data_num,fea_num]=size(X);
for i=1:data_num
    i
    tic;
    for j=i+1:data_num
        dis_num=dis_num+1;
        tmp_z=X(i,:)+X(j,:);
        non_zero=find(tmp_z~=0);
        dis(dis_num)=sum((X(i,non_zero)-X(j,non_zero)).^2./(X(i,non_zero)+X(j,non_zero)));
        %dis(dis_num)=norm(X(i,:)-X(j,:))^2/(X(i,:)+X(j,:));
    end
    toc
end


gvalue=1/mean(dis);