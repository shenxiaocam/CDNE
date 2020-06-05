function W = MyScaleSimMat(W)

degree=sum(W,2);
degree(degree==0)=1; 
  
%scale 
% D = diag(degree);
%W = D^(-1)*W;

W=W./degree;

end