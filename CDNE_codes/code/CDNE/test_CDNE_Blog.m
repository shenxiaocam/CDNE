% Created on OCT 27 18:16:14 2018
% author: SHEN xiao
% Please cite our paper:
% X. Shen, Q. Dai, S. Mao, F. Chung and K. Choi, "Network Together: Node Classification via Cross-Network Deep Network Embedding," in IEEE Transactions on Neural Networks and Learning Systems, early access, Jun. 4, 2020, doi: 10.1109/TNNLS.2020.2995483.

%% An Example Case for cross-network node classification in Blog networks %%
clear all;
addpath(genpath('../../code'));

Kstep = 3;
beta=4; %penalty on non-zero elements
alfa=4;   % weight of pairwise constraints on connected nodes
phi_S=2; % weight of pairwise constraints on labeled nodes in the source network
r_T=40; %weight of contional MMD
u_T=2; %weight of marginal MMD

%load source network
load('Blog1_30.mat');
network_S=network;
Y_S=double(group);
attrb_S=attrb;
A_S =AggTranProbMat(network_S, Kstep);% aggregated K-step transition probability matrix
PPMI_S = ComputePPMI(A_S);

%load target network
load('Blog2_30.mat');
network_T=network;
Y_T=double(group);
attrb_T=attrb;
A_T = AggTranProbMat(network_T, Kstep);
PPMI_T = ComputePPMI(A_T);


num_labels=size(Y_S,2);
num_nodes_S = size(network_S,1);
hidden_dim=[256 128]; %layer-wised hidden dimensionality setting
nnsize_S = [num_nodes_S hidden_dim];       
num_nodes_T = size(network_T,1);
nnsize_T = [num_nodes_T hidden_dim];        

%% configuration of SAE_s for source network%%
rand('state',0)
sae_S = saesetup(nnsize_S);
for i = 1: length(nnsize_S) - 1
    sae_S.ae{i}.activation_function       = 'sigm';
    sae_S.ae{i}.output                    = 'sigm';
    sae_S.ae{i}.learningRate              = 0.05;
    sae_S.ae{i}.scaling_learningRate      = 0.95;         
    sae_S.ae{i}.weightPenaltyL2           = 0.05;
end


%% configuration of SAE_t for target network%%
sae_T = saesetup(nnsize_T);
for i = 1: length(nnsize_T) - 1
    sae_T.ae{i}.activation_function       = 'sigm';
    sae_T.ae{i}.output                    = 'sigm';
    sae_T.ae{i}.learningRate              = 0.05;
    sae_T.ae{i}.scaling_learningRate      = 0.95;         
    sae_T.ae{i}.weightPenaltyL2           = 0.05;       
end


%% deep network embedding for source network
O_S=(Y_S)*(Y_S)'; % whether two nodes have common labels
O_S(find(O_S==0))=-1; %if nodes i and j are with different labels, O_S(i,j)=-1;
O_S=O_S+diag(-diag(O_S));  % set the diagonal entries as 0
disp(['SAE_s for source network embedding']);
[rep_S,rep_S_avg, Q_S]= source_SAE(sae_S,nnsize_S,PPMI_S,PPMI_S+PPMI_S',Y_S,beta,alfa,phi_S,O_S);

%% PCA on cross-network attributes
attrb_ST=[attrb_S;attrb_T];
attrb_d=128; %hidden attribute dimension
[~,rep_attrb_ST,~] = pca(full(attrb_ST),'NumComponents',attrb_d) ;
rep_attrb_S=rep_attrb_ST(1:num_nodes_S,:);
rep_attrb_T=rep_attrb_ST(num_nodes_S+1:end,:);


%% select trp of nodes in the target network which include all the label categories
trp=[0.01]; 
% trp=[0,0.005,0.01,0.03,0.05,0.07,0.1];
f1AllPer=cell(1,length(trp));
for trpindex=1:length(trp) %index in trp
    random_state=0;
    f1All=[];
    numSplit=5;   
    for randomSplit=1:numSplit
        rng('default');
        rng(random_state);
        trn=fix(trp(trpindex)*num_nodes_T);  % randomly choose trn training examples
        train_x_index=randperm(num_nodes_T,trn);
        group_T_train=zeros(size(Y_T));
        group_T_train(train_x_index,:)=Y_T(train_x_index,:);
        random_state=random_state+1;
        if trp(trpindex)~=0
            while nnz(sum(group_T_train,1))~=size(group_T_train,2)
                rng('default');
                rng(random_state);
                train_x_index=randperm(num_nodes_T,trn);
                group_T_train=zeros(size(Y_T));
                group_T_train(train_x_index,:)=Y_T(train_x_index,:);
                random_state=random_state+1;
            end
        end        
        test_x_index=setdiff([1:1:num_nodes_T], train_x_index);
        
        %% predict fuzzy labels based on PCA attributes
        X_train=[rep_attrb_S;rep_attrb_T(train_x_index,:)];
        X_test=rep_attrb_T(test_x_index,:);
        Y_train=[Y_S;Y_T(train_x_index,:)]; % full source network labels and scarce target network labels
        Y_test=Y_T(test_x_index,:);
        [Y_prob_attrb,~,~] = multi_label_class(X_train,Y_train,X_test,Y_test);
        group_T_train(test_x_index,:)=Y_prob_attrb;
        
        
        %% Deep network embedding for target network
        disp(['SAE_t for target network embedding']);
        disp([num2str(randomSplit) '-th random split with training fraction ' num2str(trp(trpindex)) ' in the target network' ]);
        [rep_T, rep_T_avg,Q_T] = target_SAE(sae_T,nnsize_T,PPMI_T,PPMI_T+PPMI_T',group_T_train,beta,alfa,Q_S,r_T,rep_S_avg,u_T);
        
        
        %% node classification trained on fully labeled nodes in source network and scarce labeled nodes in the target network
        X_train=[rep_S{end};rep_T{end}(train_x_index,:)];
        X_test=rep_T{end}(test_x_index,:);
        Y_train=[Y_S;Y_T(train_x_index,:)]; 
        Y_test=Y_T(test_x_index,:);
        [~,~,f1] = multi_label_class(X_train,Y_train,X_test,Y_test);
		f1
        f1All=[f1All;f1]; %f1 scores for each ramdon training/testing split
    end

    f1AllPer{trpindex}=f1All;
end

avgMicro=zeros(1,length(trp));
avgMacro=zeros(1,length(trp));
stdMicro=zeros(1,length(trp));
stdMacro=zeros(1,length(trp));
for j=1:length(trp) % number of training percentages
    mm=f1AllPer{1,j};
    avgMicro(1,j)=mean(mm(:,1));
    avgMacro(1,j)=mean(mm(:,2));
    stdMicro(1,j)=std(mm(:,1));
    stdMacro(1,j)=std(mm(:,2));
end
disp(['Training fraction in target network: ']);
trp
disp(['Average Micro-F1 (over 5 ramdom splits) for above training fraction: ']);
avgMicro
disp(['Average Macro-F1 (over 5 ramdom splits) for above training fraction: ']);
avgMacro
