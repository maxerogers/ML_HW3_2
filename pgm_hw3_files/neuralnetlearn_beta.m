function [best_nnet, new_neural_net, train_err, test_err] = ...
    neuralnetlearn_beta (neural_net, data, data_cv)

global alpha maxT change_thresh change_err_thresh

maxT
alpha
change_thresh 
change_err_thresh
lambda = 0
fflush(1);

n = size(data,1);

new_neural_net = neural_net;
nlayers = new_neural_net.nlayers;

nattribs = size(neural_net.layer(2).W,2)-1;

input_idx = 1:nattribs;
output_idx = (nattribs+1):size(data,2);
nout = length(output_idx);


t = 0;
max_change = realmax;
min_change_err = 1;

train_err = zeros(maxT,2);
test_err = zeros(maxT,2);

[err, err_rate] = nnet_error_beta (new_neural_net, data)
min_err = err ;
min_err_rate = err_rate ;
best_nnet = new_neural_net ;

[cv_err_best,cv_err_rate_best] = nnet_error_beta (best_nnet, data_cv) ;
[cv_err,cv_err_rate] = nnet_error_beta (new_neural_net, data_cv) ;

best_net.err = min_err;
best_net.err_rate = min_err_rate;
best_net.cv_err = cv_err_best;
best_net.cv_err_rate = cv_err_rate_best;

new_neural_net.err = err;
new_neural_net.err_rate = err_rate;
new_neural_net.cv_err = cv_err;
new_neural_net.cv_err_rate = cv_err_rate;

train_err(1,:) = [err err_rate];
test_err(1,:) = [cv_err cv_err_rate];

fflush(1);

while ((min_change_err > change_err_thresh) && (max_change > change_thresh) && (t < maxT))
  t
  fprintf('Backprop...');
  fflush (1);  

  dW = backprop_beta_vec (new_neural_net, data(:,input_idx)', data(:,output_idx)');

  fprintf('done.\n');
  fflush (1);

  fprintf('Update...');
  fflush (1);  
  max_change = 0;
  for l = nlayers:-1:2
    Delta(l).dE = dW.layer(l).vals/n;
    %delta_W = (alpha/(n*nout)) * (Delta(l).dE + lambda *
    %new_neural_net.layer(l).W);
    delta_W = alpha * (Delta(l).dE + lambda * new_neural_net.layer(l).W);
    new_neural_net.layer(l).W = new_neural_net.layer(l).W - delta_W;
    max_change = max(max_change, max(abs(delta_W(:))));
  end
  fprintf('done.\n');
  fflush (1);  
  last_err = err ;
  [err,err_rate] = nnet_error_beta (new_neural_net, data)
  min_change_err = abs ( err - last_err ) ;
  max_change
  
  [cv_err,cv_err_rate] = nnet_error_beta (new_neural_net, data_cv) ;

  new_neural_net.err = err;
  new_neural_net.err_rate = err_rate;
  new_neural_net.cv_err = cv_err;
  new_neural_net.cv_err_rate = cv_err_rate;
      
  if (err < min_err)
    min_err = err ;
    min_err_rate = err_rate ;
    best_nnet = new_neural_net ;
    
    [cv_err_best,cv_err_rate_best] = nnet_error_beta (best_nnet, data_cv) ;
    
    best_net.err = min_err;
    best_net.err_rate = min_err_rate;
    best_net.cv_err = cv_err_best;
    best_net.cv_err_rate = cv_err_rate_best;
  end
  fflush(1);

  t = t+1;

  train_err(t,:) = [err err_rate];
  test_err(t,:) = [cv_err cv_err_rate];

end

min_err
min_err_rate

train_err = train_err(1:t,:);
test_err = test_err(1:t,:);

end
