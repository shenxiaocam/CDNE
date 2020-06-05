function  [Y_prob,Y_pre,f1] = multi_label_class(X_train,Y_train,X_test,Y_test)

%% node classification trained on fully source network labeld data and scare labeled data in the target network
num_dim=size(X_train,2);
num_labels=size(Y_train,2);
Y_train=full(Y_train);
Y_test=full(Y_test);

W = zeros(num_dim+1, num_labels);
for c=1:num_labels
    %use 5-fold cross-validation on the training set to find the bestC
    %                 bestC=train(2*Y_train(:,c)-1, sparse([ones(size(X_train,1),1) X_train]), ['-C -s 0 -v 5 -e 0.01']);
    %                 bestC=bestC(1);
    %                 cmd=horzcat('-s 0 -c ', num2str(bestC,'%.8f'), ' -e 0.01');
    %
    %training labels should be full, training feature matrix should be sparse
    % change label from 1/0 to 1/-1 during training
    % add a bias ones-vector as the first column of X
    cmd='-s 0 -c 1 -e 0.01';
    model = train(2*Y_train(:,c)-1, sparse([ones(size(X_train,1),1) X_train]), [cmd]);
    W(:, c)=model.w';
end

Y_prob=sigm([ones(size(X_test,1),1) X_test]*W); %predicted probability matrix, add a bias ones-vector as the first column of X

%% final predicted label matrix
Y_pre=zeros(size(Y_prob));
for ii=1:size(Y_pre,1)
    k=sum(Y_test(ii,:));% find out how many labels should be predicted
    [~,originalpos] = sort(Y_prob(ii,:), 'descend' );
    index = originalpos(1:k); %the index of the labels with the highest predicted probability for node i
    Y_pre(ii,index)=1;
end

[micro,macro]=F1score(Y_test,Y_pre);
f1=[micro,macro];

end