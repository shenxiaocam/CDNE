function nn = saeff_t(nn, x, y,beta,L_A_T,alfa_T,Y_T_batch,Q_S,r_T,rep_S_avg,u_T)
%NNFF performs a feedforward pass

%% add more penalty to non-zero elements for autoencoder %%

n = nn.n;
m = size(x, 1);

x = [ones(m,1) x];
nn.a{1} = x;

%feedforward pass
for i = 2 : n-1
    switch nn.activation_function
        case 'sigm'
            % Calculate the unit's outputs (including the bias term)
            nn.a{i} = sigm(nn.a{i - 1} * nn.W{i - 1}');
        case 'tanh_opt'
            nn.a{i} = tanh_opt(nn.a{i - 1} * nn.W{i - 1}');
    end
    
    %dropout
    if(nn.dropoutFraction > 0)
        if(nn.testing)
            nn.a{i} = nn.a{i}.*(1 - nn.dropoutFraction);
        else
            nn.dropOutMask{i} = (rand(size(nn.a{i}))>nn.dropoutFraction);
            nn.a{i} = nn.a{i}.*nn.dropOutMask{i};
        end
    end
    
    %calculate running exponential activations for use with sparsity
    if(nn.nonSparsityPenalty>0)
        nn.p{i} = 0.99 * nn.p{i} + 0.01 * mean(nn.a{i}, 1);
    end
    
    %Add the bias term
    nn.a{i} = [ones(m,1) nn.a{i}];
end
switch nn.output
    case 'sigm'
        nn.a{n} = sigm(nn.a{n - 1} * nn.W{n - 1}');
    case 'tanh_opt'
        nn.a{n} = tanh_opt(nn.a{n - 1} * nn.W{n - 1}');
    case 'linear'
        nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
    case 'softmax'
        nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
        nn.a{n} = exp(bsxfun(@minus, nn.a{n}, max(nn.a{n},[],2)));
        nn.a{n} = bsxfun(@rdivide, nn.a{n}, sum(nn.a{n}, 2));
end

%error and loss
nn.e = y - nn.a{n};

%% add more penalty to non-zero input elements for autoencoder %%
if(beta~=1)
    nonzero_index=find(y~=0);
    nn.e(nonzero_index)= nn.e(nonzero_index)*beta;
end

switch nn.output
    case {'sigm', 'linear','tanh_opt'}
        nn.L = 1/2 * sum(sum((nn.e).^ 2)) / m;
    case 'softmax'
        nn.L = -sum(sum(y .* log(nn.a{n}))) / m;
end


%% add pairwise constraints J_2 %%
H=nn.a{2}(:,2:end); %hidden representation learned from autoencoder
if (alfa_T>0)
    nn.L= nn.L+ (alfa_T/m)*trace(H'*L_A_T*H);
end

%% add pairwise constraints J_3 %%
if (r_T>0)
    C_Y_T_batch = diag(1./sum(Y_T_batch,1));
    C_Y_T_batch(find(C_Y_T_batch==Inf))=0;
    Nor_Y_T_batch=Y_T_batch*C_Y_T_batch; % normalized Y_T_batch by column summation
    Q_T=Nor_Y_T_batch'*H;
    I_T_batch=zeros(size(Q_T));
    for ii=1:size(I_T_batch,1)
        if( nnz(Q_S(ii,:))>0)&&(nnz(Q_T(ii,:))>0)
            I_T_batch(ii,:)=ones(1,size(I_T_batch,2));
        end
    end
    nn.L= nn.L+(r_T/2)*sumsqr(I_T_batch.*(Q_T-Q_S));
end

%% J_4 MMD %%
if(u_T>0)
   nn.L= nn.L+(u_T/2)*sumsqr(rep_S_avg-mean(H,1)) ;
end

end
