function [dW, delta] = backprop_beta_vec (neural_net, input, output)

nlayers = neural_net.nlayers;
new_neural_net = neural_net;

[net_output, y] = forwardprop_beta (neural_net, input);

delta.layer(nlayers).vals = feval(neural_net.ds, net_output) .* ...
    feval(neural_net.dE, net_output, output);

for l = nlayers:-1:3
  x = y.layer(l-1).output(2:end,:);
  d = delta.layer(l).vals;
  delta.layer(l-1).vals = feval (neural_net.ds, x) .* ...
      (neural_net.layer(l).W(:,2:end)' * d);
end

for l = nlayers:-1:2
  x = y.layer(l-1).output;
  d = delta.layer(l).vals;
  dW.layer(l).vals = d * x';
end

end
