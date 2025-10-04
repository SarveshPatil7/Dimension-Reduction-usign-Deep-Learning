
function train_dfns()
    Nd_vals = [10, 50, 100, 200];
    Ns_multipliers = [1, 5, 10, 20];
    latent_dim = 10;

    for Nd = Nd_vals
        for m = Ns_multipliers
            Ns = Nd * m;

            x_file = sprintf('train_X_Nd%d_Ns%d.csv', Nd, Ns);
            theta_file = sprintf('latent_Nd%d_Ns%d.csv', Nd, Ns);

            % Load data
            x_train = readmatrix(x_file);            % Ns x Nd
            theta_train = readmatrix(theta_file);    % Ns x latent_dim

            % Transpose for nn(.) format: Nd x Ns
            x_train = x_train';
            theta_train = theta_train';

            n = min(ceil(Ns / 5), 12); 
            net = fitnet(n, 'trainlm');

            % Set linear activation
            for i = 1:length(net.layers)
                net.layers{i}.transferFcn = 'purelin';
            end

            % Training parameters
            net.trainParam.epochs = 500;
            net.trainParam.mu_max = 1e11;
            net.trainParam.max_fail = 6;
            if Ns < 200
                net.trainParam.min_grad = 1e-11;
            else
                net.trainParam.min_grad = 1e-6;
            end
            net.trainParam.showWindow = true;
            net.performFcn = 'mse';

            % Train the DFN
            fprintf('\nTraining DFN for Nd = %d, Ns = %d\n', Nd, Ns);
            net = train(net, x_train, theta_train);

            % Save trained network
            save(sprintf('dfn_net_Nd%d_Ns%d.mat', Nd, Ns), 'net');

            %pause(5);
        end
    end
end
