function evaluate_dfn_r_values()
    Nd_vals = [10, 50, 100, 200];
    Ns_multipliers = [1, 5, 10, 20];
    latent_dim = 5;

    results = [];

    for Nd = Nd_vals
        for m = Ns_multipliers
            Ns = Nd * m;

            % Load model and data
            dfn_file = sprintf('dfn_net_Nd%d_Ns%d.mat', Nd, Ns);
            x_file = sprintf('train_X_Nd%d_Ns%d.csv', Nd, Ns);
            theta_file = sprintf('latent_Nd%d_Ns%d.csv', Nd, Ns);

            if exist(dfn_file, 'file') && exist(x_file, 'file') && exist(theta_file, 'file')
                load(dfn_file, 'net');
                x_train = readmatrix(x_file)';
                theta_train = readmatrix(theta_file)';

                % Predict latent variables
                y_hat = net(x_train);

                % Compute R values for each latent dim
                R = zeros(1, latent_dim);
                for i = 1:latent_dim
                    R(i) = corr(y_hat(i,:)', theta_train(i,:)');
                end

                % Mean R
                R_mean = mean(R);
                fprintf('Nd = %d, Ns = %d | R_mean = %.4f\\n', Nd, Ns, R_mean);

                results = [results; Nd, Ns, R];
            else
                fprintf('Missing files for Nd = %d, Ns = %d\\n', Nd, Ns);
            end
        end
    end

    % Save results to CSV
    %writematrix(results, 'dfn_r_values.csv');
end
