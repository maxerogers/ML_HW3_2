function [nnet] = make_nnet_beta (nunits, s, ds, E, dE)

nnet.nlayers = length(nunits);

for l = 2:nnet.nlayers
  nnet.layer(l).W = 2*rand(nunits(l), nunits(l-1)+1)-1;
  %nnet.layer(l).W = zeros(nunits(l), nunits(l-1)+1);
end

nnet.s = s;
nnet.ds = ds;
nnet.E = E;
nnet.dE = dE;

nnet.err = Inf;
nnet.cv_err = Inf;

nnet.err_rate = 1;
nnet.cv_err_rate = 1;

end