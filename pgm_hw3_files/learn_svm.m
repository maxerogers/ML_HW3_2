

load 'D.dat'
load 'D_cv.dat'
data = D ;
data_cv = D_cv ;

global kernel_type maxT minT rel_chg_thresh


kernel_type = 'poly';
maxT = 5000;
minT = 15;

rel_chg_thresh = 1e-3;

nclass = 10;

X = data(:,1:(end-1)) ;
Y_tmp = data(:,end) ;
[m,n] = size( X ) ;


X_cv = data_cv(:,1:(end-1)) ;
Y_cv_tmp = data_cv(:,end) ;
m_cv = size( X_cv , 1 ) ;

f_val = zeros ( m_cv , nclass ) ;
err_rate = ones ( nclass , 1) ;
f_val_train = zeros ( m , nclass ) ;
err_rate_train = ones ( nclass , 1) ;

for d = 1:nclass
  printf ( 'processing digit %d...\n', d-1  ) ; fflush (1) ; 
  pos_dig = d-1 ;

  pos_ex = ( Y_tmp == pos_dig ) ;
  neg_ex = ~pos_ex ;
  Y = -ones( m , 1 ) ;
  Y(pos_ex) = 1 ;
   
  D = [X Y];

  % svm(d).sv = sort([(find(pos_ex))(ceil(rand*sum(pos_ex))) ...
  %                   (find(neg_ex))(ceil(rand*sum(neg_ex)))],'ascend');
  % svm(d).alpha = zeros(m,1) ;
  % svm(d).alpha(svm(d).sv) = 1 ;
  % svm(d).sv = (1:m)';
  % svm(d).alpha = ones(m,1) ;
  % svm(d).alpha(pos_ex) = 1e4 / sum(pos_ex) ;
  % svm(d).alpha(neg_ex) = 1e4 / sum(neg_ex) ;
  svm(d).sv = [];
  svm(d).alpha = zeros(m,1) ;
  svm(d).b = 0 ;
  svm(d).data = D ;
  svm(d).last_sv = [];
  svm(d).last_alpha = zeros(m,1) ;
  svm(d).last_b = 0 ;
  svm(d).last_data = D ;
  svm(d).C = 1e4 ;
  %svm(d).kernel = kernel ;
  svm(d).cv_err = 0 ;
  svm(d).err = 0 ;
 
  new_svm = svmlearn_beta_v5 ( svm(d) , D ) ; 
   
  svm(d) = new_svm ;
  
  b = new_svm.b ;
  sv_idx_list = new_svm.sv ;
  alpha_sv = new_svm.alpha ;
  X_sv = X(sv_idx_list,:) ;
  Y_sv = Y(sv_idx_list) ;
    
  f_val_train(:,d) = svmeval ( svm(d) , X ) ;
  Y_pred_train = sign ( f_val_train(:,d) ) ;
  train_err = mean( single(( Y_pred_train .* Y ) < 0) ) 
  err_rate_train(d) = train_err ;
  svm(d).err = train_err ;
  
  pos_ex_cv = ( Y_cv_tmp == pos_dig ) ;
  neg_ex_cv = ~pos_ex_cv ;
  Y_cv = -ones ( m_cv , 1 ) ;
  Y_cv( pos_ex_cv ) = 1 ;
  
  f_val(:,d) = svmeval ( svm(d) , X_cv ) ;
  Y_pred = sign ( f_val(:,d) ) ;
  cv_err = mean( single(( Y_pred .* Y_cv ) < 0) ) 
  err_rate(d) = cv_err ;
  svm(d).cv_err = cv_err ;

  prop_sv = length( sv_idx_list ) / m 
  
  fflush (1) ; 
end

for d=1:nclass
  fprintf(1, 'Error rate for digit %d SVM = %f ; train = %f\n', d-1, ...
          err_rate(d), err_rate_train(d));
end

[f_opt_train, Y_pred_tmp_train ] = max ( f_val_train' ) ;

Y_pred_train = Y_pred_tmp_train' - 1 ;

train_err = mean ( single(Y_pred_train ~= Y_tmp ) )

[f_opt, Y_pred_tmp ] = max ( f_val' ) ;

Y_pred = Y_pred_tmp' - 1 ;

cv_err = mean ( single(Y_pred ~= Y_cv_tmp ) )

fprintf(1, 'Error rate for final SVM = %f ; train = %f\n', cv_err, train_err);

load 'D_trial.dat'
data_trial = D_trial;

X_trial = data_trial(:,1:(end-1)) ;
m_trial = size ( data_trial , 1 ) ;

f_val_trial = zeros ( m_trial , nclass ) ;
 
for d = 1:nclass
  f_val_trial(:,d) = svmeval ( svm(d) , X_trial ) ;
end

[f_opt_trial, Y_pred_tmp_trial ] = max ( f_val_trial' ) ;

Y_pred_trial = Y_pred_tmp_trial' - 1 

