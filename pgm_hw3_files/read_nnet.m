function new_nnet = read_nnet ( nnet )

new_nnet = nnet;

for l = 2:nnet.nlayers
  eval( ['load W_' num2str(l-1) '_'  num2str(l-2) ...
         '.dat' ] ) ;
  eval( ['new_nnet.layer(l).W = W_' num2str(l-1) ...
         '_'  num2str(l-2) ' ; '] ) ;
end


end