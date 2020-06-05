%% Calculate Positive Pointwise Mutual Information Matrix %%

function PPMI = ComputePPMI(M)
M=M+diag(-diag(M));  % set the diagonal entries as 0

M = MyScaleSimMat(M);

[p, q] = size(M);
assert(p==q, 'M must be a square matrix!'); 

col = sum(M,1);
col(col==0)=1;
PPMI = log( (p*M) ./col); %PMI
IdxNan = isnan(PPMI);
PPMI(IdxNan) = 0;
PPMI(PPMI<0)=0; %max(PPMI,0)


end