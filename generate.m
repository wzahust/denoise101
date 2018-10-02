function generate
%生成数据
sigma1=10;sigma2=10;
x_base=zeros(100,64,64);
x=zeros(30000,64,64);y=zeros(30000,64,64);
for i=1:100
    u1=unidrnd(64);u2=unidrnd(64);
    for m=1:64
        for n=1:64
        x_base(i,m,n)=(100/(sigma1*sigma2))*exp(-(0.5*(n-u2)^2)/(sigma1^2)-(0.5*(m-u1)^2)/(sigma2^2));
        end
    end
end
for j=1:100
    for p=1:300
        q=(j-1)*300+p;
        x(q,:,:)=x_base(j,:,:)+0.1*normrnd(0,1,1,64,64);
        y(q,:,:)=x_base(j,:,:);
    end
end

%画图
y_temp=zeros(64,64);x_temp=zeros(64,64);k=325;
for a=1:64
    for b=1:64
        y_temp(a,b)=y(k,a,b);
        x_temp(a,b)=x(k,a,b);
    end
end
a=1:64;b=1:64;
surf(a,b,y_temp(a,b));
surf(a,b,x_temp(a,b));
end