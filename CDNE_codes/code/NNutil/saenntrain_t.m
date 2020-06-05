function [nn, L]  = saenntrain_t(nn, train_x, train_y, opts, beta,A_T, alfa,Q_S,Y_T_train, r,rep_S_avg,u,val_x, val_y)
%NNTRAIN trains a neural net
% [nn, L] = nnff(nn, x, y, opts) trains the neural network nn with input x and
% output y for opts.numepochs epochs, with minibatches of size
% opts.batchsize. Returns a neural network nn with updated activations,
% errors, weights and biases, (nn.a, nn.e, nn.W, nn.b) and L, the sum
% squared error for each training minibatch.

assert(isfloat(train_x), 'train_x must be a float');
assert(nargin == 12 || nargin == 14,'number ofinput arguments must be 12 or 14')

loss.train.e               = [];
loss.train.e_frac          = [];
loss.val.e                 = [];
loss.val.e_frac            = [];
opts.validation = 0;
if nargin == 14
    opts.validation = 1;
end

fhandle = [];
if isfield(opts,'plot') && opts.plot == 1
    fhandle = figure();
end

m = size(train_x, 1);

batchsize = opts.batchsize;
numepochs = opts.numepochs;

numbatches = m / batchsize;

%---update here by Shaosheng---%
numbatches = floor(numbatches);
%------------------------------%

assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');

L = zeros(numepochs*numbatches,1);
n = 1;
for epoch = 1 : numepochs

    kk = randperm(m);
    for l = 1 : numbatches
        batch_x = train_x(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        batch_x_index=kk((l - 1) * batchsize+1: l * batchsize); % the index of the nodes in this batch
       
        % get the PPMI matrix between the samples in each batch
        A_T_batch=zeros(batchsize,batchsize); %whether two nodes in the batch are connected    
        for a=1:batchsize
            for b=1:batchsize
                A_T_batch(a,b)=A_T(batch_x_index(a),batch_x_index(b)) ; 
            end
        end
        D_A_T = diag(sum(A_T_batch,2)); % the degree matrix of adjacency matrix
        L_A_T = D_A_T - A_T_batch;   % the laplace matrix of adjacency matrix
       
        % get the labels of nodes in each batch
        Y_T_batch=zeros(batchsize, size(Y_T_train,2));   
        for i=1:batchsize
            Y_T_batch(i,:)=Y_T_train(batch_x_index(i),:);
        end
       
        
        %Add noise to input (for use in denoising autoencoder)
        if(nn.inputZeroMaskedFraction ~= 0)
            batch_x = batch_x.*(rand(size(batch_x))>nn.inputZeroMaskedFraction);
        end
        
        batch_y = train_y(kk((l - 1) * batchsize + 1 : l * batchsize), :);
       
        nn = saeff_t(nn, batch_x, batch_y,beta,L_A_T,alfa,Y_T_batch,Q_S,r,rep_S_avg,u);
        nn = saebp_t(nn,batch_x,beta,L_A_T,alfa,Y_T_batch,Q_S,r,rep_S_avg,u);
        nn = nnapplygrads(nn);
        
        L(n) = nn.L;
        
        n = n + 1;
    end
    


%     if opts.validation == 1
%         loss = nneval(nn, loss, train_x, train_y, val_x, val_y);
%         str_perf = sprintf('; Full-batch train mse = %f, val mse = %f', loss.train.e(end), loss.val.e(end));
%     else
%         loss = nneval(nn, loss, train_x, train_y);
%         str_perf = sprintf('; Full-batch reconstruction errors = %f', loss.train.e(end));
%     end

    if (mod(epoch, 10) == 0)   
        disp(['epoch ' num2str(epoch) '/' num2str(opts.numepochs) '. Target loss is ' num2str(mean(L((n-numbatches):(n-1))))]);
    end
    nn.learningRate = nn.learningRate * nn.scaling_learningRate;
    
%   if (mod(epoch, 10) == 0) % update figure after each 10 epoches
%       if ishandle(fhandle)
%           nnupdatefigures(nn, fhandle, loss, opts, epoch);
%       end
%   end
  
  
end
end

