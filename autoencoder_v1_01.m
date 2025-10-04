function train_all_autoencoders()
    Nd_vals = [10, 50, 100, 200];
    Ns_multipliers = [1, 5, 10, 20];

    latent_dim = 5;
    epochs = 500;
    lr = 0.05;

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

                % Train the autoencoder
                fprintf('\nTraining autoencoder for Nd = %d, Ns = %d\n', Nd, Ns);
                [encoder_weights, decoder_weights, theta_t] = autoencoder_2layer(Dt, latent_dim, epochs, lr);

                % Save training data
                writematrix(theta_t, sprintf('latent_Nd%d_Ns%d.csv', Nd, Ns));
                writematrix(encoder_weights.W1, sprintf('encoder_W1_Nd%d_Ns%d.csv', Nd, Ns));
                writematrix(encoder_weights.b1, sprintf('encoder_b1_Nd%d_Ns%d.csv', Nd, Ns));
                writematrix(encoder_weights.W2, sprintf('encoder_W2_Nd%d_Ns%d.csv', Nd, Ns));
                writematrix(encoder_weights.b2, sprintf('encoder_b2_Nd%d_Ns%d.csv', Nd, Ns));
                writematrix(decoder_weights.W3, sprintf('decoder_W3_Nd%d_Ns%d.csv', Nd, Ns));
                writematrix(decoder_weights.b3, sprintf('decoder_b3_Nd%d_Ns%d.csv', Nd, Ns));
                writematrix(decoder_weights.W4, sprintf('decoder_W4_Nd%d_Ns%d.csv', Nd, Ns));
                writematrix(decoder_weights.b4, sprintf('decoder_b4_Nd%d_Ns%d.csv', Nd, Ns));

            else
                fprintf('Warning: training file %s not found.\n', x_file, y_file);
            end
        end
    end
end


function [encoder_weights, decoder_weights, theta_t] = autoencoder_2layer(Dt, latent_dim, epochs, lr)

    % Dimensions
    [Ns, input_dim] = size(Dt);
    hidden_dim = 32;

    % Activation functions
    sigmoid = @(x) 1 ./ (1 + exp(-x));
    sigmoid_deriv = @(x) sigmoid(x) .* (1 - sigmoid(x));

    % -- Weight Initialization --
    W1 = randn(input_dim, hidden_dim) * 0.01;
    b1 = zeros(1, hidden_dim);

    W2 = randn(hidden_dim, latent_dim) * 0.01;
    b2 = zeros(1, latent_dim);

    W3 = randn(latent_dim, hidden_dim) * 0.01;
    b3 = zeros(1, hidden_dim);

    W4 = randn(hidden_dim, input_dim) * 0.01;
    b4 = zeros(1, input_dim);

    % Training loop
    for epoch = 1:epochs
        % ---------- Forward Pass ----------
        Z1 = Dt * W1 + b1;
        A1 = sigmoid(Z1);          % Encoder Layer 1

        Z2 = A1 * W2 + b2;
        H = sigmoid(Z2);           % Latent Layer

        Z3 = H * W3 + b3;
        A3 = sigmoid(Z3);          % Decoder Layer 1

        Z4 = A3 * W4 + b4;
        Dt_hat = sigmoid(Z4);               % Reconstructed Output

        % ---------- Loss ----------
        loss = mean(sum((Dt_hat - Dt).^2, 2));

        % ---------- Backpropagation ----------
        dZ4 = 2 * (Dt_hat - Dt) / Ns;             % Linear output

        dW4 = A3' * dZ4;
        db4 = sum(dZ4, 1);

        dA3 = dZ4 * W4';
        dZ3 = dA3 .* sigmoid_deriv(Z3);

        dW3 = H' * dZ3;
        db3 = sum(dZ3, 1);

        dH = dZ3 * W3';
        dZ2 = dH .* sigmoid_deriv(Z2);

        dW2 = A1' * dZ2;
        db2 = sum(dZ2, 1);

        dA1 = dZ2 * W2';
        dZ1 = dA1 .* sigmoid_deriv(Z1);

        dW1 = Dt' * dZ1;
        db1 = sum(dZ1, 1);

        % ---------- Update Weights ----------
        W1 = W1 - lr * dW1; b1 = b1 - lr * db1;
        W2 = W2 - lr * dW2; b2 = b2 - lr * db2;
        W3 = W3 - lr * dW3; b3 = b3 - lr * db3;
        W4 = W4 - lr * dW4; b4 = b4 - lr * db4;

        % ---------- Print Progress ----------
        if mod(epoch, 50) == 0 || epoch == 1
            fprintf('Epoch %d/%d | Loss: %.6f\n', epoch, epochs, loss);
        end
    end

    % Output
    encoder_weights = struct('W1', W1, 'b1', b1, 'W2', W2, 'b2', b2);
    decoder_weights = struct('W3', W3, 'b3', b3, 'W4', W4, 'b4', b4);
    theta_t = sigmoid(sigmoid(Dt * W1 + b1) * W2 + b2);  % final encoded layer
end


