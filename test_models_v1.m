function test_models()
    Nd_vals = [10, 50, 100, 200];
    Ns_multipliers = [1, 5, 10, 20];

    for Nd = Nd_vals
        % Load test file
        x_test_file = sprintf('test_X_Nd%d.csv', Nd);
        y_test_file = sprintf('test_Y_Nd%d.csv', Nd);

        x_test = readmatrix(x_test_file);  % Ns x Nd
        y_true = readmatrix(y_test_file);  % Ns x 1

        for m = Ns_multipliers
            Ns = Nd * m;
            dfn_file = sprintf('dfn_net_Nd%d_Ns%d.mat', Nd, Ns);
            gp_file = sprintf('gp_model_Nd%d_Ns%d.mat', Nd, Ns);

            fprintf('\nTesting models for Nd = %d, Ns = %d\n', Nd, Ns);

            % Load models
            load(dfn_file, 'net');
            load(gp_file, 'gprMdl');

            % Predict latent variables
            theta_pred = net(x_test');         % latent_dim x Ns
            theta_pred = theta_pred';          % transpose to Ns x latent_dim

            % Predict final outputs
            y_pred = predict(gprMdl, theta_pred);

            % Mean Square Error
            mse_val = mean((y_pred - y_true).^2);
            fprintf('MSE: %.4f\n', mse_val);

        end
    end
end
