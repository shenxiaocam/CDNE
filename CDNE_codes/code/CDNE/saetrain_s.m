% Created on OCT 27 18:16:14 2018
% author: SHEN xiao
% Please cite our paper:
% X. Shen, Q. Dai, S. Mao, F. Chung and K. Choi, "Network Together: Node Classification via Cross-Network Deep Network Embedding," in IEEE Transactions on Neural Networks and Learning Systems, early access, Jun. 4, 2020, doi: 10.1109/TNNLS.2020.2995483.


function sae = saetrain_s(sae, x, beta,A_S, alfa_S,phi_S,O_S)


for i = 1 : numel(sae.ae)
    disp(['Training SAE_s ' num2str(i) '/' num2str(numel(sae.ae))]);
    opts.numepochs =100;
    if i==1
        opts.batchsize = 100;       
    else
        opts.batchsize = 50;    
        alfa_S=alfa_S/2;  %weight of pairwise constraints on connected nodes at deep layer of SAE_s
        phi_S= 0; % weight of pairwise constraints on common labeled nodes at deep layer of SAE_s
    end
   
        
    sae.ae{i} = saenntrain_s(sae.ae{i}, x, x, opts,beta,A_S, alfa_S,phi_S,O_S);
    
    t = nnff(sae.ae{i}, x, x);
    x = t.a{2};
    %remove bias term
    x = x(:,2:end);
    
end

end
