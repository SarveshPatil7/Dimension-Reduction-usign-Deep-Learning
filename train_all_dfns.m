
function train_dfns()
    Nd_vals = [10, 50, 100, 200];
    Ns_multipliers = [1, 5, 10, 20];
    %Nd_vals = [200];
    %Ns_multipliers = [20];
    latent_dim = 5;

    for Nd = Nd_vals
        for m = Ns_multipliers
            Ns = Nd * m;

            % File paths
            x_file = sprintf('train_X_Nd%d_Ns%d.csv', Nd, Ns);
            theta_file = sprintf('latent_Nd%d_Ns%d.csv', Nd, Ns);

            % Check if files exist
            if exist(x_file, 'file') && exist(theta_file, 'file')
                % Load data
                x_train = readmatrix(x_file);            % Ns x Nd
                theta_train = readmatrix(theta_file);    % Ns x latent_dim

                % Transpose for MATLAB NN format: features x samples
                x_train = x_train';
                theta_train = theta_train';

                % Create DFN with 2 hidden layers of 20 neurons each
                net = fitnet([20, 20], 'trainlm');

                % Use sigmoid (logsig) for all layers including output
                for i = 1:length(net.layers)
                    net.layers{i}.transferFcn = 'logsig';
                end

                % Set training parameters
                net.trainParam.epochs = 500;
                net.trainParam.min_grad = 1e-6;
                net.trainParam.showWindow = true;
                net.performFcn = 'mse';

                % Train the DFN
                fprintf('\nTraining DFN for Nd = %d, Ns = %d\n', Nd, Ns);
                net = train(net, x_train, theta_train);

                % Save trained network
                save(sprintf('dfn_net_Nd%d_Ns%d.mat', Nd, Ns), 'net');
            else
                fprintf('Warning: missing file %s or %s\n', x_file, theta_file);
            end
        end
    end
end
