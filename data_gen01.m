clc; clear all;

Nd_vals = [10, 50, 100, 200];
Ns_vals = [1, 5, 10,20];
Nt = 1000;

function y = griewank(x, Nd)
idx = [1:Nd]
t1 = sum(x.^2,2) / 4000
t2 = prod(cos(x./ sqrt(idx)), 2)
y = t1 - t2 + 1
end

for Nd = Nd_vals
    x_test = lhsdesign(1,Nd)
    y_test = griewank(x_test, Nd)

    writematrix(x_test, sprintf('test_X_Nd%d.csv', Nd));
    writematrix(y_test, sprintf('test_Y_Nd%d.csv', Nd));

    for Ns = Ns_vals * Nd
        x_train = lhsdesign(Ns,Nd)
        y_train = griewank(x_train, Nd)

        writematrix(x_train, sprintf('train_X_Nd%d_Ns%d.csv', Nd, Ns));
        writematrix(y_train, sprintf('train_Y_Nd%d_Ns%d.csv', Nd, Ns));
    end
end
