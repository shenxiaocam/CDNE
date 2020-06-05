function A = AggTranProbMat(G, step)
%return the aggregated transition probability matrix

G = MyScaleSimMat(G);        %scale by row
A_k = G;
A=G; %aggregated transition probability matrix
for k = 2:step
    A_k=A_k*G;            %A_k indicates A^k
    A=A+A_k/k; % weighting the k-step co-occurrence matrix by 1/k
end
A=sparse(A);

end