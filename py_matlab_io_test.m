% import py.py.py_matlab_io_test.*
% x=[1,6,8];x=py.tuple(x);
% y=[1,7,9];y=py.tuple(y);
% v=py.tuple({x,y});
% v2=v;
% vs=py.tuple({v,v2});
% res=mysum(vs)

% xx=py.tuple(v);
% xx=py.list({1,2,3});
% res=mysum(xx)
%?: res=cell(res); res=cellfun(@double,res);
%more:
%cn.mathworks.com/help/matlab/matlab_external/pass-data-to-matlab-from-python.html
%cn.mathworks.com/help/matlab/matlab_external/handling-data-returned-from-python.html

% a=[[1,2];[2,3];[4,5]];
% x(1,:,:)=a(:,:);
% x(2,:,:)=[[6,7];[8,9];[10,11]];
a=[[1;2],[2;3],[4;5]];
x(1,:,:)=a(:,:);
x(2,:,:)=[[6;7],[8;9],[10;11]];
save x.mat x
