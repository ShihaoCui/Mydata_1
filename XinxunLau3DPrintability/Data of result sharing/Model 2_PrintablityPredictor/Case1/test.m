load('y_prediction.mat')
load('y_test.mat')
err = y_prediction-y_test;
err = sign(err-0.5)+1;
err = err/2;

err1 = 1-sum(err',1)

sum(err1)
x_test_selected = x_test([1,2,6,7,9,16],:);
y_test_selected = y_test([1,2,6,7,9,16],:);

save slected1 x_test_selected  y_test_selected