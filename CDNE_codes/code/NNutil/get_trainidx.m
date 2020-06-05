%% An Example Case %%
clear all;
addpath(genpath('../../code'));


%load target network
load('PPI2.mat');
network_T=network;
Y_T=group;
attrb_T=attrb;
num_nodes_T = size(network_T,1);


%% select trp of nodes in the target network which include all the label categories
% trp=[0.01];
trp=[0.005,0.01,0.03,0.05,0.07,0.1];
trainindexAllPer=cell(1,length(trp));

for trpindex=1:length(trp) %index in trp
    random_state=0;
    trainindexAll=[];

    
    for randomSplit=1:5
        rng('default');
        rng(random_state);
        trn=fix(trp(trpindex)*num_nodes_T);  % randomly choose trn training examples
        train_x_index=randperm(num_nodes_T,trn);
        group_T_train=zeros(size(Y_T));
        group_T_train(train_x_index,:)=Y_T(train_x_index,:);
        random_state=random_state+1;
        
        %% if not all the class have training examples, resample       
        while nnz(sum(group_T_train,1))~=size(group_T_train,2)
            rng('default');
            rng(random_state);
            train_x_index=randperm(num_nodes_T,trn);
            group_T_train=zeros(size(Y_T));
            group_T_train(train_x_index,:)=Y_T(train_x_index,:);
            random_state=random_state+1;
        end

        test_x_index=setdiff([1:1:num_nodes_T], train_x_index);

        
        trainindexAll=[trainindexAll;train_x_index];
    end
    
    trainindexAllPer{trpindex}= trainindexAll;
    
end

save('PPI2_trainindex.mat','trainindexAllPer','attrb','network','group');
