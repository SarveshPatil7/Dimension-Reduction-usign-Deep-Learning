function train_all_gp_surrogates()
    %Nd_vals = [10, 50, 100, 200];
    %Ns_multipliers = [1, 5, 10, 20];
    Nd_vals = [10, 50, 100];
    Ns_multipliers = [1, 5, 10, 20];

    for Nd = Nd_vals
        for m = Ns_multipliers
            Ns = Nd * m;

            % File paths
            theta_file = sprintf('latent_Nd%d_Ns%d.csv', Nd, Ns);
            y_file = sprintf('train_Y_Nd%d_Ns%d.csv', Nd, Ns);

            if exist(theta_file, 'file') && exist(y_file, 'file')
                theta_train = readmatrix(theta_file);  % Ns x latent_dim
                y_train = readmatrix(y_file);          % Ns x 1

                % Fit Gaussian Process Regression model
                sigma0 = 1;  % Initial noise standard deviation guess
                latent_dim = size(theta_train, 2);
                kparams0 = ones(1, latent_dim);  % Initial length scale per latent dimension

                fprintf('\nTraining GP surrogate for Nd = %d, Ns = %d\n', Nd, Ns);
                tic;
                gprMdl = fitrgp(theta_train, y_train, ...
                    'KernelFunction', 'squaredexponential', ...
                    'KernelParameters', kparams0, ...
                    'Sigma', sigma0);
                elapsed_time = toc;
                
                % Evaluate training performance
                y_pred_train = predict(gprMdl, theta_train);
                mse_train = mean((y_pred_train - y_train).^2);
                R2_train = 1 - sum((y_pred_train - y_train).^2) / sum((y_train - mean(y_train)).^2);

                % Report
                fprintf('Training time: %.2f seconds\n', elapsed_time);
                fprintf('MSE (train): %.4f | R² (train): %.4f\n', mse_train, R2_train);

                % Save the GP model
                save(sprintf('gp_model_Nd%d_Ns%d.mat', Nd, Ns), 'gprMdl');
            else
                fprintf('Warning: missing file %s or %s\n', theta_file, y_file);
            end
        end
    end
end
