function nn = saebp_s(nn,x,beta,L_A_S, alfa_S,L_O_S,phi_S)
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
        if (alfa_S~=0)
            d{i}=d{i}+[zeros(size(d{i},1),1) (alfa_S*(L_A_S+L_A_S')*H).*d_act_1]; %pairewise constraints devirations on connected nodes
        end

        %% partial derivatives of J_3,  pairewise constraints on common labeled nodes  
        if (phi_S~=0)
           d{i}=d{i}+[zeros(size(d{i},1),1) (phi_S*(L_O_S+L_O_S')*H).*d_act_1]; %pairewise constraints devirations on nodes with same labels
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
