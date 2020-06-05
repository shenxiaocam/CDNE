function nn = nnapplygrads(nn)
%NNAPPLYGRADS updates weights and biases with calculated gradients
% nn = nnapplygrads(nn) returns an neural network structure with updated
% weights and biases
    
    for i = 1 : (nn.n - 1)
            
        if(nn.weightPenaltyL2>0)
            dW = nn.dW{i} + nn.weightPenaltyL2 * [zeros(size(nn.W{i},1),1) nn.W{i}(:,2:end)];
        else
            dW = nn.dW{i};
        end
        
%         if (i==1)
%             Y=nn.a{2}(:,2:end); % hidden representations learned from autoencoder
%             dYW=((Y.*(1-Y))')*nn.a{1}(:,2:end);
%             size(dW)
%             size(2*(laplace+laplace')*Y*dYW)
%             dW=dW+2*(laplace+laplace')*Y*dYW;
%         end
        
        dW = nn.learningRate * dW;
        
        if(nn.momentum>0)
            nn.vW{i} = nn.momentum*nn.vW{i} + dW;
            dW = nn.vW{i};
        end
            
        nn.W{i} = nn.W{i} - dW;
    end
end
