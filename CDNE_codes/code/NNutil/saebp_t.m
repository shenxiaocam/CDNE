function nn = saebp_t(nn,x,beta,L_A_T,alfa,Y_T_batch,Q_S,r_T,rep_S_avg,u_T)
%NNBP performs backpropagation returns an neural network structure with updated weights

n = nn.n;
sparsityError = 0;
switch nn.output
    case 'sigm'
        d{n} = - nn.e.* (nn.a{n} .* (1 - nn.a{n}));
    case {'softmax','linear'}
        d{n} = - nn.e;
    case 'tanh_opt'
        d{n} = - nn.e.*(1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn.a{n}.^2));
end

%% add more penalty to non-zero input elements for autoencoder %%
if(beta~=1)
    nonzero_index=find(x~=0);
    d{n}(nonzero_index)=d{n}(nonzero_index)*beta;
end

for i = (n - 1) : -1 : 2
    % Derivative of the activation function
    switch nn.activation_function
        case 'sigm'
            d_act = nn.a{i} .* (1 - nn.a{i});
        case 'tanh_opt'
            d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn.a{i}.^2);
    end
    
    if(nn.nonSparsityPenalty>0)
        pi = repmat(nn.p{i}, size(nn.a{i}, 1), 1);
        sparsityError = [zeros(size(nn.a{i},1),1) nn.nonSparsityPenalty * (-nn.sparsityTarget ./ pi + (1 - nn.sparsityTarget) ./ (1 - pi))];
    end
    
    % Backpropagate first derivatives
    if i+1==n % in this case in d{n} there is not the bias term to be removed
        d{i} = (d{i + 1} * nn.W{i} + sparsityError) .* d_act; % Bishop (5.56)
    else % in this case in d{i} the bias term has to be removed
        d{i} = (d{i + 1}(:,2:end) * nn.W{i} + sparsityError) .* d_act;
    end
    
    if(nn.dropoutFraction>0)
        d{i} = d{i} .* [ones(size(d{i},1),1) nn.dropOutMask{i}];
    end
    
    
    
    %pairewise constraints devirations
    if i==2
        switch nn.activation_function
            case 'sigm'
                d_act_1 = nn.a{i}(:,2:end) .* (1 - nn.a{i}(:,2:end));
            case 'tanh_opt'
                d_act_1 = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn.a{i}(:,2:end).^2);
        end
        
        H=nn.a{i}(:,2:end); 
       %% partial derivatives of J_2,  pairewise constraints on connected nodes 
        if (alfa>0)
            d{i}=d{i}+[zeros(size(d{i},1),1) (alfa*(L_A_T+L_A_T')*H).*d_act_1]; 
        end
        
      %% partial derivatives of J_3,  pairewise constraints on cross-network alignment  
       n_t=size(x,1);
       if(r_T>0)
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
   
           DJH= Nor_Y_T_batch*(I_T_batch.*I_T_batch.*(Q_T-Q_S)); 
           d{i}=d{i}+[zeros(size(d{i},1),1) n_t*r_T*DJH.*d_act_1]; 
        end
        
       %% partial derivatives of J_4,  MMD
        if(u_T>0)
            d{i}=d{i}+[zeros(size(d{i},1),1) u_T*n_t*ones(n_t,1)*(mean(H,1)-rep_S_avg).*d_act_1]; 
        end
        
    end    
end



for i = 1 : (n - 1)
    if i+1==n
        nn.dW{i} = (d{i + 1}' * nn.a{i}) / size(d{i + 1}, 1);
    else
        nn.dW{i} = (d{i + 1}(:,2:end)' * nn.a{i}) / size(d{i + 1}, 1);
    end
end

end
