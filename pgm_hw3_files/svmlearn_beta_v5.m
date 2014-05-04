function new_svm = svmlearn_beta_v5 (svm, data)

global kernel_type maxT minT rel_chg_thresh

minT
maxT
rel_chg_thresh
fflush(1) ;

n = size(data,1);
alpha = zeros(n,1);

new_svm = svm;
sv = svm.sv;
alpha(sv) = svm.alpha(sv);
b = svm.b;
C = svm.C;

alpha = alpha / C ;

y = data(:,end);
X = data(:,1:(end-1));
%K = svm.kernel(X,X);
K = feval(['kernel_' kernel_type],X,X);

pos_ex = y > 0 ;
neg_ex = y < 0 ;

pos_ex_list = find(pos_ex);
neg_ex_list = find(neg_ex);

g_val = 0

y_K_y = K .* (y * y') ;

%all(eig(y_K_y) > 0)

slack = 1 - y_K_y * alpha * C - y * b ;
slack_tmp=slack;
slack_tmp(slack < 0)  = 0 ;
[max_slack,max_k] = max(slack_tmp);

max_slack

%active_set_idx = logical(ones(n,1)); %(slack_tmp >= (max_slack - 0.01)) ;

%active_set = find(active_set_idx) ;
%num_active = length(active_set) 
fflush(1);

% slack_tmp = slack ;
% slack_tmp(slack < 0)  = 0 ;

% [max_slack,max_k] = max(slack_tmp);

% max_slack

% [max_slack_pos,max_k_pos] = max(slack_tmp(pos_ex));
% [max_slack_neg,max_k_neg] = min(-slack_tmp(neg_ex));

% active_k_pos = pos_ex_list(slack_tmp(pos_ex) == max_slack_pos);
% active_k_neg = neg_ex_list(-slack_tmp(neg_ex) == max_slack_neg);

% active_set = [active_k_pos ; active_k_neg] 
% active_set_idx = zeros (n,1);
% active_set_idx(active_set) = logical(1);

num_change = n ;
old_num_change = 0 ;
sv_idx = alpha > 0;
old_sv_idx = zeros(n,1);
max_change = 1;

avg_rel_change = 1 ;

best_g_val = g_val;
best_alpha = alpha;
best_b = b;

for t = 1:maxT
  num_sv = length(sv)
  sum_alpha_y = alpha' * y
  g_val_old = g_val;
  g_val = -0.5 * alpha' * y_K_y * alpha + sum(alpha)/C 
  if (g_val > best_g_val)
    best_g_val = g_val;
    best_alpha = alpha;
    best_b = b;
  end
  g_val_change = g_val - g_val_old 
  if (g_val_old == 0)
    if (abs(g_val_change) == 0)
      g_val_rel_change = 0 
    else
      g_val_rel_change = 1 
    end
  else
    g_val_rel_change = abs(g_val_change / g_val_old)
  end
  avg_rel_change = (1-1/t) * avg_rel_change + (1/t) * g_val_rel_change 
  fflush(1) ;
  if ((t > minT) && 
    (g_val_change / g_val_old < rel_chg_thresh))
    %((avg_rel_change < rel_chg_thresh) || (g_val_change / g_val_old < -1e-5)))
    % (abs(g_val_change/g_val_old) < rel_chg_thresh)) ...
    %  && (all(sv_idx == old_sv_idx))
    %&& (max_change < 1e-10)) 
    break; end
  grad = (1/C - y_K_y * alpha) ;
  %grad(~active_set_idx) = 0 ;
  %grad = grad - y .* b / C ;
  y_mean_y_grad = y .* mean ( y .* grad ) ;
  proj_grad = grad - y_mean_y_grad ;
  old_num_change = num_change ;
  num_change = n ;
  stay_0 = zeros(n,1);
  stay_1 = zeros(n,1);
   count = 0 ;
  %change = active_set_idx ;
  change = logical(ones(n,1)) ;
  do
   count = count + 1 ;
  stay_0 = stay_0 | (( alpha == 0 ) & ( proj_grad <= 0 )) ;
  stay_1 = stay_1 | (( alpha == 1 ) & ( proj_grad >= 0 )) ;
  change = change & ~ ( stay_0 | stay_1 ) ;
  new_proj_grad = proj_grad ;
  new_proj_grad( ~ change ) = 0;
  
  y_mean_y_grad = y .* mean ( y .* new_proj_grad ) ;
  proj_grad = new_proj_grad - y_mean_y_grad ;
  proj_grad( ~ change ) = 0 ;
  stay_0 = stay_0 | (( alpha == 0 ) & ( proj_grad <= 0 )) ;
  stay_1 = stay_1 | (( alpha == 1 ) & ( proj_grad >= 0 )) ;
  change = change & ~ ( stay_0 | stay_1 ) ;
  new_proj_grad = proj_grad ;
  new_proj_grad( ~ change ) = 0;  
  proj_grad = new_proj_grad;
  old_num_change = num_change ;
  num_change = sum(change) ;
   fflush(1);
  until (num_change-old_num_change == 0)
  
  num_change
  fflush(1);
  
  if (num_change == 0) break; end 
  
  pos_grad = proj_grad > 0;
  neg_grad = proj_grad < 0;
  lbound = Inf ;
  ubound = -Inf ;
  if (any(pos_grad))
    lbound = min(lbound,min(-alpha(pos_grad)./proj_grad(pos_grad)));
    ubound = max(ubound,max((1-alpha(pos_grad))./proj_grad(pos_grad)));
  end
  if (any(neg_grad))
    lbound = min(lbound,min((1-alpha(neg_grad))./proj_grad(neg_grad)));
    ubound = max(ubound,max(-alpha(neg_grad)./proj_grad(neg_grad)));
  end

  lambda = max(lbound,min((proj_grad' * grad)/(proj_grad' * y_K_y * ...
                                                proj_grad),ubound))

  if (lambda == 0), break; end
  
  %tmp_alpha_2 = alpha + lambda * proj_grad ;
  %g_val_tmp = -0.5 * tmp_alpha_2' * y_K_y * tmp_alpha_2 + sum(tmp_alpha_2)/C
      
  %g_val_change_tmp = g_val_tmp - g_val
  
  %beta_tmp = tmp_alpha_2' * y
  
  tmp_alpha = min(max(alpha + lambda * proj_grad, zeros(n,1)), ...
                  repmat(C,n,1));
  beta = tmp_alpha' * y ;

  if (abs(beta) > 0)
    sv = (tmp_alpha > 0) ;
    pos_sv = sv  & pos_ex ;
    neg_sv = sv & neg_ex ;
    
    num_pos_sv = sum(pos_sv) ;
    num_neg_sv = sum(neg_sv) ;
    
    neg_sv_beta_pos = neg_sv & (beta > 0);
    pos_sv_beta_neg = pos_sv & (beta < 0);
    
    if ((num_pos_sv > 0) && (num_neg_sv > 0))
      shift_tmp_alpha = tmp_alpha - beta * y .* (neg_sv_beta_pos/num_neg_sv ...
                                                 + pos_sv_beta_neg/num_pos_sv) ...
          ;
    % elseif ((num_pos_sv == 0) && (num_neg_sv > 0))
    %   shift_tmp_alpha = tmp_alpha - beta * y .* (neg_sv_beta_pos/num_neg_sv) ...
    %       ;
    % elseif ((num_pos_sv > 0) && (num_neg_sv == 0))
    %   shift_tmp_alpha = tmp_alpha - beta * y .* (pos_sv_beta_neg/num_pos_sv) ...
    %       ;
    else
      shift_tmp_alpha = tmp_alpha * 0;
    end
    
    if (any(shift_tmp_alpha > 1))
      new_tmp_alpha = shift_tmp_alpha / max(shift_tmp_alpha);
    else
      new_tmp_alpha = shift_tmp_alpha ;
    end
  else
    new_tmp_alpha = tmp_alpha ;
  end
  max_change = max ( abs ( new_tmp_alpha - alpha ) ) 
  alpha = new_tmp_alpha ;

  old_sv_idx = sv_idx ;
  sv = find(alpha);
  sv_idx = alpha > 0 ;
  
  b = ((1-alpha(sv))' * (1 - y_K_y(sv,sv) * alpha(sv) * C))/((1-alpha(sv))' ...
                                                     * y(sv))

  %true_sv = find(sv_idx & (alpha < 1)) ;

  %b = mean(y(true_sv).*(1- y_K_y(true_sv,:) * alpha * C))
  
  slack = 1 - y_K_y * alpha * C - y * b ;
  slack_tmp = slack ;
  slack_tmp(slack < 0)  = 0 ;
  
  [max_slack,max_k] = max(slack_tmp);
  
  max_slack

  %active_set_idx = (slack >= - 0.01) ;

  %active_set = find(active_set_idx) ;
  %num_active = length(active_set) 
  fflush(1);

  % [max_slack_pos,max_k_pos] = max(slack_tmp(pos_ex));
  % [max_slack_neg,max_k_neg] = min(-slack_tmp(neg_ex));

  % active_k_pos = pos_ex_list(max_k_pos);
  % active_k_neg = neg_ex_list(max_k_neg);
  
  % active_set = [active_k_pos ; active_k_neg] 
  % active_set_idx = active_set_idx * 0 ;
  % active_set_idx(active_set) = logical(1);

end

g_val = -0.5 * alpha' * y_K_y * alpha + sum(alpha)/C

last_sv = find(alpha);
new_svm.last_sv = last_sv;
new_svm.last_alpha = alpha(last_sv) * C;


new_svm.last_b = b;
new_svm.last_data = data(last_sv,:);

sv = find(best_alpha);
new_svm.sv = sv;
new_svm.alpha = best_alpha(sv) * C;


new_svm.b = best_b;
new_svm.data = data(sv,:);

end