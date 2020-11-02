img=imread('lena512.tif');
[M,N]=size(img);
training_set=reshape(img,M*N,1);
shape=size(img);
res=zeros(7,shape(1),shape(2));
for n = 1:7
    nbit=2^n;
    [partition, codebook] = lloyds(training_set,nbit);
    for i=1:shape(1)
        for j=1:shape(2)
            if (img(i,j)>0) && (img(i,j)<partition(1))
                res(n,i,j)=codebook(1);
            else
                for m=2:nbit-1
                    if img(i,j)>=partition(m-1) && img(i,j)<partition(m)
                        res(n,i,j)=codebook(m);
                        break
                    end
                    res(n,i,j)=codebook(nbit);

                end
            end
        end
    end
end
img_scale=zeros(7,M,N);
for n =1:7
    img_scale(n,:,:)=u_q(img,n);
end
for n = 1:7
    a=squeeze(res(n,:,:));
    mean= sum(sum((double(img) - a).^2)) / numel(img);
    mean_lm(n)=mean;
    b=squeeze(img_scale(n,:,:));
    mean=sum(sum((double(img) - b).^2)) / numel(img);
    mean_uq(n)=mean;
end

x1=1:7;
figure
plot(x1,log(mean_lm),x1,log(mean_uq))
title('MSE vs number of bits')
xlabel('Bits')
ylabel('log(MSE)')
legend('Lloyd-Max quantization','uniform quantization')


img_re = histeq(img,256);
[M,N]=size(img);
training_set=reshape(img_re,M*N,1);
shape=size(img_re);
res=zeros(7,shape(1),shape(2));
for n = 1:7
    nbit=2^n;
    [partition, codebook] = lloyds(training_set,nbit);
    for i=1:shape(1)
        for j=1:shape(2)
            if (img_re(i,j)>=0) && (img_re(i,j)<partition(1))
                res(n,i,j)=codebook(1);
            else
                for m=2:nbit-1
                    if img_re(i,j)>=partition(m-1) && img_re(i,j)<partition(m)
                        res(n,i,j)=codebook(m);
                        break
                    end
                    res(n,i,j)=codebook(nbit);

                end
            end
        end
    end
end

img_scale=zeros(7,M,N);
for n =1:7
    img_scale(n,:,:)=u_q(img_re,n);
end
for n = 1:7
    a=squeeze(res(n,:,:));
    mean= sum(sum((double(img_re) - a).^2)) / numel(img_re);
    mean_lm(n)=mean;
    b=squeeze(img_scale(n,:,:));
    mean=sum(sum((double(img_re) - b).^2)) / numel(img_re);
    mean_uq(n)=mean;
end

x1=1:7;
figure
plot(x1,log(mean_lm),x1,log(mean_uq))
title('MSE vs number of bits')
xlabel('Bits')
ylabel('log(MSE)')
legend('Lloyd-Max quantization','uniform quantization')

