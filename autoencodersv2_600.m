function train_all_autoencoders()
    Nd_vals = [10, 50, 100, 200];
    Ns_multipliers = [1, 5, 10, 20];

    latent_dim = 10;
    epochs = 2000;

    for Nd = Nd_vals
        for m = Ns_multipliers
            Ns = Nd * m;

            % Load training data
            x_file = sprintf('train_X_Nd%d_Ns%d.csv', Nd, Ns);
            y_file = sprintf('train_Y_Nd%d_Ns%d.csv', Nd, Ns);

            x_train = readmatrix(x_file);
            y_train = readmatrix(y_file);
            Dt = [x_train, y_train];

            % Call training function
            fprintf('\nTraining autoencoder (SCG) for Nd = %d, Ns = %d\n', Nd, Ns);
            [net, theta_t] = train_autoencoder_scg(Dt, latent_dim, epochs);

            % Save the files
            writematrix(theta_t, sprintf('latent_Nd%d_Ns%d.csv', Nd, Ns));
            save(sprintf('autoencoder_net_Nd%d_Ns%d.mat', Nd, Ns), 'net');
        end
    end
end

function [net, theta_t] = train_autoencoder_scg(Dt, latent_dim, max_epochs)
    input_dim = size(Dt, 2);

    % Create Scaled Conjugate Gradient Descent based feedforward autoencoder
    net = feedforwardnet([64, latent_dim, 64], 'trainscg');   % 64,nz,64

    % Set linear activation 
    for i = 1:length(net.layers)
        net.layers{i}.transferFcn = 'purelin';
    end

    % Training parameters
    net.trainParam.epochs = max_epochs;
    net.trainParam.min_grad = 1e-6;
    net.trainParam.max_fail = 15;  % instead of default 6
    net.trainParam.showWindow = true;
    net.performFcn = 'mse';

    % Train autoencoder
    [net, tr] = train(net, Dt', Dt');
    
    % Extract latent variables
    theta_t = net(Dt')';
    R_vals = corr(Dt, theta_t);
    R_mean = mean(diag(R_vals));
    
    %pause(5);
end
