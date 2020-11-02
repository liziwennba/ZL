function [s_bit] = u_q(img, s)
shape=size(img);
res=zeros(shape);
ss=8-s;
temp=2^ss;
m=2^s;
delta=uint8(255/m);
for i=1:shape(1)
    for j=1:shape(2)
        a=img(i,j);
        b=round(a/delta);
        if ( b ==m)
            b = b-1;
        end
        if b < 0
            b = 0;
        end
        res(i,j)=b*delta;
    end
end
s_bit=uint8(res);


%temp=128
%(129/128)*256-1

