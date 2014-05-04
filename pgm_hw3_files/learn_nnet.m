
global alpha maxT change_thresh change_err_thresh

change_thresh = 1e-4;
change_err_thresh = 1e-6;

load 'D.dat'
load 'D_cv.dat'
nunits = [1024 64 10];
maxT = 200;
alpha = 21;
data = D ;
data_cv = D_cv ;

X = data(:,1:(end-1)) ;
Y_tmp = data(:,end) ;
m = size ( X , 1 ) ;
Y = ( repmat ( 0:9 , m , 1 ) == repmat ( Y_tmp , 1 , 10 ) ) ;
D_new = [ X Y ] ;


nnet = make_nnet_beta ( nunits , 'sigmoid' , 'deriv_sigmoid' , ...
                        'squared_error' , 'deriv_squared_error' ) ;

X_cv = data_cv(:,1:(end-1)) ;
Y_tmp = data_cv(:,end) ;
m = size ( X_cv , 1 ) ;
Y_cv = ( repmat ( 0:9 , m , 1 ) == repmat ( Y_tmp , 1 , 10 ) ) ;
D_cv_new = [ X_cv Y_cv ] ;

[best_nnet, last_nnet, train_err, test_err] = neuralnetlearn_beta ( nnet , D_new , D_cv_new ) ;

[best_nnet.err best_nnet.cv_err best_nnet.err_rate best_nnet.cv_err_rate]
[last_nnet.err last_nnet.cv_err last_nnet.err_rate last_nnet.cv_err_rate]


output_nnet ( best_nnet , 'best_nnet' ) ;
output_nnet ( last_nnet , 'last_nnet' ) ;


fid = fopen ( 'train_err.dat' , 'w' ) ;
fprintf ( fid , '%f\n' , train_err(:,1) ) ;
fclose ( fid ) ;

fid = fopen ( 'test_err.dat' , 'w' ) ;
fprintf ( fid , '%f\n' , test_err(:,1) ) ;
fclose ( fid ) ;

fid = fopen ( 'train_err_rate.dat' , 'w' ) ;
fprintf ( fid , '%f\n' , train_err(:,2) ) ;
fclose ( fid ) ;

fid = fopen ( 'test_err_rate.dat' , 'w' ) ;
fprintf ( fid , '%f\n' , test_err(:,2) ) ;
fclose ( fid ) ;

