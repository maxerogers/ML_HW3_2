function [f,K] = svmeval (svm, query)

global kernel_type

sv_X = svm.data(:,1:(end-1));
sv_y = svm.data(:,end);

K = feval(['kernel_' kernel_type], sv_X , query );
alpha_y = svm.alpha .* sv_y;

f = alpha_y' * K + svm.b ;

end