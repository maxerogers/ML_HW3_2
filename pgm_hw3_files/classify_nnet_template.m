
load 'X.dat'
nunits = [ 64 10 ]   

nnet = make_nnet_beta ( nunits , 'sigmoid' , 'deriv_sigmoid' , ...
                        'squared_error' , 'deriv_squared_error' ) ;

nnet = read_nnet ( nnet ) ;

output = forwardprop_beta (nnet, X');

[ output_val , output_idx ] = max ( output ) ;

fid = fopen ( 'Y.dat' , 'w' ) ;
fprintf ( fid , '%d\n' , output_idx'-1 ) ;
fclose ( fid ) ; 


