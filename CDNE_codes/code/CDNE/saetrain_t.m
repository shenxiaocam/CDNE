% Created on OCT 27 18:16:14 2018
% author: SHEN xiao
% Please cite our paper:
% X. Shen, Q. Dai, S. Mao, F. Chung and K. Choi, "Network Together: Node Classification via Cross-Network Deep Network Embedding," in IEEE Transactions on Neural Networks and Learning Systems, early access, Jun. 4, 2020, doi: 10.1109/TNNLS.2020.2995483.

function sae = saetrain_t(sae, x, beta,A_T, alfa_T, Q_S,Y_T_train, r_T,rep_S_avg,u_T)


for i = 1 : numel(sae.ae)
    disp(['Training SAE_t ' num2str(i) '/' num2str(numel(sae.ae))]);
    opts.numepochs =100;
	
    if i==1
        opts.batchsize = 100;  
    else
        opts.batchsize = 50; 
        alfa_T=alfa_T/2;  %weight of pairwise constraints on connected nodes at the deep layer of SAE_t
        r_T=r_T/2; % weight of conditional MMD at the deep layer of SAE_t
        u_T=u_T/2; %weight of marginal MMD at the deep layer of SAE_t
    end
    
    
    sae.ae{i} = saenntrain_t(sae.ae{i}, x, x, opts,beta,A_T, alfa_T, Q_S{i},Y_T_train, r_T,rep_S_avg{i},u_T);
    
    t = nnff(sae.ae{i}, x, x);
    x = t.a{2};
    %remove bias term
    x = x(:,2:end);
    
end

end
