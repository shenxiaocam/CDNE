
% Created on OCT 27 18:16:14 2018
% author: SHEN xiao
% Please cite our paper:
% Shen, X.; Dai, Q.; Mao, S.; Chung, F.-l.; and Choi, K.-S. 2020. Network Together: Node Classification via Cross network Deep Network Embedding. IEEE Transactions on Neural Networks and Learning Systems.


function  [rep_T, rep_T_avg,Q_T]= target_SAE(sae_T,nnsize,network_T,PPMI_T,group_T_train,beta,alfa_T,Q_S,r_T,rep_S_avg,u_T)

sae_T = saetrain_t(sae_T,  network_T, beta,  PPMI_T, alfa_T, Q_S, group_T_train,r_T,rep_S_avg,u_T);  
rep_T = GenRep(network_T, sae_T, nnsize);   % each layer of hidden representations for the target network

%% compute average feature vector representations of target network
rep_T_avg=cell(size(rep_T));
for layeri=1:length(rep_T_avg)
    rep_T_avg{layeri}=mean(rep_T{layeri},1);
end

%% compute class-conditional average feature vector representations of target network
Q_T=cell(size(rep_T));
C_Y_T = diag(1./sum(group_T_train,1));
C_Y_T(find(C_Y_T==Inf))=0;
for rlayer=1:length(rep_T) 
    Q_T{rlayer}=(group_T_train*C_Y_T)'*rep_T{rlayer};
end


end

