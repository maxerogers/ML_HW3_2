function [y,y_rate] = nnet_error_beta (nnet, data)

n = size(data,1);

nlayers = nnet.nlayers;

nattribs = size(nnet.layer(2).W,2)-1;

input_idx = 1:nattribs;
output_idx = (nattribs+1):size(data,2);

output = forwardprop_beta (nnet, data(:,input_idx)');
err = mean((output - data(:,output_idx)').^2,1);

y = mean(err);
%/length(output_idx);

y_rate = 1-mean(all(data(:,output_idx)' == (output == repmat(max(output),size(output,1),1)),1));

end
