function output_nnet ( nnet , fname_prefix )

for l = 2:nnet.nlayers
  fid = fopen ( [ fname_prefix '_W_' num2str(l-1) '_'  num2str(l-2) '.dat' ], 'w' ) ...
        ;
  matformat = repmat( ['%f '] , 1 , size(nnet.layer(l).W,2) ) ;
  matformat = [matformat '\n'];
  fprintf ( fid , matformat , nnet.layer(l).W' ) ;
  fclose ( fid ) ;
end

fid = fopen ( [ fname_prefix '_err_final.dat' ] , 'w' ) ;
fprintf ( fid , '%f %f %f %f\n' , nnet.err , nnet.cv_err , nnet.err_rate ...
          , nnet.cv_err_rate ) ;
fclose ( fid ) ;



end