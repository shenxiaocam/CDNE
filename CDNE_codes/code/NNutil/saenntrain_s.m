function [nn, L]  = saenntrain_s(nn, train_x, train_y, opts, beta,A_S, alfa_S, phi_S,O_S,val_x, val_y)
%NNTRAIN trains a neural net
% [nn, L] = nnff(nn, x, y, opts) trains the neural network nn with input x and
% output y for opts.numepochs epochs, with minibatches of size
% opts.batchsize. Returns a neural network nn with updated activations,
% errors, weights and biases, (nn.a, nn.e, nn.W, nn.b) and L, the sum
% squared error for each training minibatch.

assert(isfloat(train_x), 'train_x must be a float');
assert(nargin == 9 || nargin == 11,'number of input arguments must be 9 or 11')

loss.train.e               = [];
loss.train.e_frac          = [];
loss.val.e                 = [];
loss.val.e_frac            = [];
opts.validation = 0;
if nargin == 11
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
        
        % compute the adjacency matrix A_S_batch, and common label matrix O_S_batch between the samples in each batch
        batch_x_index=kk((l - 1) * batchsize+1: l * batchsize);
        A_S_batch=zeros(batchsize,batchsize); %whether two nodes are connected
        O_S_batch=zeros(batchsize,batchsize); %whether two nodes have common labels  
        
        for a=1:batchsize
            for b=1:batchsize
                A_S_batch(a,b)=A_S(batch_x_index(a),batch_x_index(b)) ; 
                 O_S_batch(a,b)=O_S(batch_x_index(a),batch_x_index(b)) ; 
            end
        end
        
        D_A_S = diag(sum(A_S_batch,2)); % the degree matrix of adjacency matrix
        L_A_S = D_A_S - A_S_batch;   % the laplace matrix of adjacency matrix
        
        O_S_P=max(O_S_batch,0);
        O_S_N=-min(O_S_batch,0);
        D_O_S_P = diag(sum(O_S_P,2)); % the degree matrix of O_S_P
        D_O_S_N = diag(sum(O_S_N,2)); % the degree matrix of O_S_N
        L_O_S_P= D_O_S_P- O_S_P; %laplace matrix of O_S_P
        L_O_S_N= D_O_S_N- O_S_N; %laplace matrix of O_S_N      
        L_O_S =  L_O_S_P - L_O_S_N; 
     
        %Add noise to input (for use in denoising autoencoder)
        if(nn.inputZeroMaskedFraction ~= 0)
            batch_x = batch_x.*(rand(size(batch_x))>nn.inputZeroMaskedFraction);
        end
        
        batch_y = train_y(kk((l - 1) * batchsize + 1 : l * batchsize), :);
       
        nn = saeff_s(nn, batch_x, batch_y,beta,L_A_S, alfa_S, L_O_S,phi_S);
        nn = saebp_s(nn,batch_x,beta,L_A_S, alfa_S, L_O_S,phi_S);
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
        disp(['epoch ' num2str(epoch) '/' num2str(opts.numepochs) '. Source loss is ' num2str(mean(L((n-numbatches):(n-1))))]);
    end
	
    nn.learningRate = nn.learningRate * nn.scaling_learningRate;
    
%   if (mod(epoch, 10) == 0) % update figure after each 10 epoches
%       if ishandle(fhandle)
%           nnupdatefigures(nn, fhandle, loss, opts, epoch);
%       end
%   end
  
  
end
end

