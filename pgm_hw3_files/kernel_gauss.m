function val = kernel_gauss(x,y)

inv_var = 0.001;

val = exp(inv_var*(x*y'-(repmat(sum(x.^2,2),1,size(y,1)) + repmat(sum(y.^2,2)',size(x,1),1))/2));

