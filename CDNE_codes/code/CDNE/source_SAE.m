% Created on OCT 27 18:16:14 2018
% author: SHEN xiao
% Please cite our paper:
% X. Shen, Q. Dai, S. Mao, F. Chung and K. Choi, "Network Together: Node Classification via Cross-Network Deep Network Embedding," in IEEE Transactions on Neural Networks and Learning Systems, early access, Jun. 4, 2020, doi: 10.1109/TNNLS.2020.2995483.
function  [rep_S,rep_S_avg, Q_S,sae_S]= source_SAE(sae_S,nnsize,network_S,PPMI_S,Y_S,beta,alfa_S,phi_S,O_S)

sae_S = saetrain_s(sae_S, network_S, beta,  PPMI_S, alfa_S,phi_S,O_S);  
rep_S = GenRep(network_S, sae_S, nnsize);   % each layer of hidden representations for the source network

%% compute average feature vector representations of source network
rep_S_avg=cell(size(rep_S));
for layeri=1:length(rep_S_avg)
    rep_S_avg{layeri}=mean(rep_S{layeri},1);
end

%% compute class-conditional average feature vector representations of source network
C_Y_S = diag(1./sum(Y_S,1));
C_Y_S(find(C_Y_S==Inf))=0;
Q_S=cell(size(rep_S));
for rlayer=1:length(rep_S) 
    Q_S{rlayer}=(Y_S*C_Y_S)'*rep_S{rlayer};
end


end
