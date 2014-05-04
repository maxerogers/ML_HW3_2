function [output, y] = forwardprop_beta (neural_net, input)

nlayers = neural_net.nlayers;

y.nlayers = nlayers;
m = size(input,2);
y.layer(1).output = [repmat(-1,1,m); input];

for l = 2:nlayers
  % W(l) such that w_{ji} is the weight of connection between unit
  % i in layers l-1 and unit j in layer l.
  z = neural_net.layer(l).W * y.layer(l-1).output;
  y.layer(l).output = [repmat(-1,1,m); feval(neural_net.s, z)];
end

output = y.layer(nlayers).output(2:end,:);

end
