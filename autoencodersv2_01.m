function train_all_autoencoders()
    %Nd_vals = [10, 50, 100, 200];
    %Ns_multipliers = [1, 5, 10, 20];
    Nd_vals = [10, 50, 100, 200];
    Ns_multipliers = [1, 5, 10, 20];

    latent_dim = 20;
    epochs = 2000;

    for Nd = Nd_vals
        for m = Ns_multipliers
            Ns = Nd * m;

            % Load training data
            x_file = sprintf('train_X_Nd%d_Ns%d.csv', Nd, Ns);
            y_file = sprintf('train_Y_Nd%d_Ns%d.csv', Nd, Ns);

            if exist(x_file, 'file') && exist(y_file, 'file')
                x_train = readmatrix(x_file);
                y_train = readmatrix(y_file);
                Dt = [x_train, y_train];

                % Train using SCG-based autoencoder
                fprintf('\nTraining autoencoder (SCG) for Nd = %d, Ns = %d\n', Nd, Ns);
                [net, theta_t] = train_autoencoder_scg(Dt, latent_dim, epochs);

                % Save latent variables
                writematrix(theta_t, sprintf('latent_Nd%d_Ns%d.csv', Nd, Ns));

                % Save full network
                save(sprintf('autoencoder_net_Nd%d_Ns%d.mat', Nd, Ns), 'net');
            else
                fprintf('Warning: training file %s or %s not found.\n', x_file, y_file);
            end
        end
    end
end

function [net, theta_t] = train_autoencoder_scg(Dt, latent_dim, max_epochs)
    input_dim = size(Dt, 2);

    % Create SCG-trained feedforward autoencoder
    % 64,nz,64
    % 64,nz,64
    % 64,32,nz,32,64
    % 64,nz,64
    net = feedforwardnet([64, latent_dim, 64], 'trainscg');

    % Set linear activations for all layers
    for i = 1:length(net.layers)
        net.layers{i}.transferFcn = 'logsig';
    end

    % Training parameters
    net.trainParam.epochs = max_epochs;
    net.trainParam.min_grad = 1e-6;
    net.trainParam.max_fail = 15;  % instead of default 6

    net.trainParam.showWindow = true;
    net.performFcn = 'mse';

    % Train with identity output (autoencoder)
    net = train(net, Dt', Dt');
    
    % Generate latent features (forward pass)
    theta_t = net(Dt')';
    R_vals = corr(Dt, theta_t);
    R_mean = mean(diag(R_vals));
    [net, tr] = train(net, Dt', Dt');

    %disp(['Stopped due to: ', tr.stop]);
    %disp(['Final gradient: ', num2str(tr.gradient)]);
    fprintf('R_mean = %.4f\n\n', R_mean);
    fprintf('MSE (train): %.4f | MSE (val): %.4f | MSE (test): %.4f\n', ...
    tr.best_perf, tr.best_vperf, tr.best_tperf);
    %pause(5);
end
