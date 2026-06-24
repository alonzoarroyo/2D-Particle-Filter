% 2D Particle Filter 
function [particles, Neff_history, resample_flags,threshold, x_true_history, x_est_history] = particle_filter_2d(T,N, Q, R, threshold_factor)
% Time steps
dt = 0.1;

x_true = [0.1; -0.1]; % initial true states
state_dim = 2; % dimensions 
% Initialize filter arrays
particles = zeros(state_dim, N); 
particles_update = zeros(state_dim, N); 
P_w = zeros(1, N); % Particle weights

% Initialize output arrays
x_true_history = zeros(state_dim,T+1);
x_true_history(:,1) = x_true; 
x_est_history = zeros(state_dim,T+1);
x_est = mean(particles, 2); 
x_est_history(:,1) = x_est;
Neff_history = zeros(1, T);
resample_flags = zeros(1, T); % Track when resampling occurred

% Measurement Noise terms
R_inv = inv(R);
R_det_term = sqrt((2*pi)^state_dim * det(R));

% Initialize particle distribution as Gaussian around true initial value
initial_cov = [2, 0.5; 0.5, 2]; % Initial covariance
for i = 1:N
    particles(:, i) = x_true + chol(initial_cov)' * randn(state_dim, 1);
end

%% Main Particle Filter Loop 
for t = 1:T
    % Update True System Model
    % Nonlinear motion model with process noise
    x_true(1) = 0.5*x_true(1) + 25*x_true(1)/(1 + x_true(1)^2) +  8*cos(1.2*(t)*dt) + 0.1*x_true(2) + sqrt(Q(1,1))*randn;
    x_true(2) = 0.5*x_true(2) + 20*x_true(2)/(1 + x_true(2)^2) +  6*sin(0.8*(t)*dt) + 0.1*x_true(1) + sqrt(Q(2,2))*randn;
    
    % Noisy Measurement
    z_true = x_true + chol(R)' * randn(state_dim, 1);
    
    % Store actual measurement
    %z_meas = [z_meas, z_true];
    
    % PREDICTION
    % Propagate particles through 2D state model
    for i = 1:N
        % Extract current particle
        x_part = particles(1, i);
        y_part = particles(2, i);
        
        % Apply 2D motion model with process noise
        process_noise = chol(Q)' * randn(state_dim, 1);
        
        particles_update(1, i) = 0.5*x_part + 25*x_part/(1 + x_part^2) + ...
                                 8*cos(1.2*(t-1)*dt) + 0.1*y_part + process_noise(1);
        
        particles_update(2, i) = 0.5*y_part + 20*y_part/(1 + y_part^2) + ...
                                 6*sin(0.8*(t-1)*dt) + 0.1*x_part + process_noise(2);
    end
    
    % UPDATE weights 
    for i = 1:N
        % Calculate weight using multivariate Gaussian likelihood
        innovation = z_true - particles_update(:, i);   
        P_w(i) = exp(-0.5 * innovation' * R_inv * innovation) / R_det_term;
    end
   
    % Normalize weights to form probability distribution
    P_w = P_w ./ sum(P_w);
    Neff = 1 / sum(P_w.^2);
    Neff_history(t) = Neff;
    threshold = N * threshold_factor;
    
    if Neff < threshold
        resample_flags(t) = 1;
        cum_weights = cumsum(P_w);
        
        % Generate systematic sampling points
        u = ([0:N-1] + rand()) / N;
        
        % Resampling
        indices = zeros(1, N);
        j = 1;
        for i = 1:N
            while u(i) > cum_weights(j)
                j = j + 1;
                if j > N
                    j = N;
                    break;
                end
            end
            indices(i) = j;
        end

        particles = particles_update(:, indices);
        % Reset weights to uniform after resampling
        P_w = ones(1, N) / N;
        
    else
        % No resampling
        resample_flags(t) = 0;
        particles = particles_update;
    end
    
    %Estimate state 
    x_est = zeros(state_dim, 1);
    for i = 1:N
        x_est = x_est + particles(:, i) * P_w(i);
    end
  
    % Store history
    x_true_history(:,t+1) =  x_true;
    x_est_history(:,t+1) =  x_est;
end
end


Q = [1, 0.1; 0.1, 0.8]; % Process covariance
R = [1, 0; 0, 1]; % easurement covariance

N = 50; % Number of particles per state
T = 100;

threshold_factor = 0.5; % Resampling threshold

[particles, Neff_history, resample_flags, threshold, x_true_history, x_est_history] = particle_filter_2d(T,N, Q, R, threshold_factor);


%% Plots
figure();
clf;

% 2D Tracking Results
subplot(2,3,1);
plot(x_true_history(1,:), x_true_history(2,:), 'b-', 'LineWidth', 2);
hold on;
plot(x_est_history(1,:), x_est_history(2,:), 'r--', 'LineWidth', 2);
scatter(particles(1,:), particles(2,:), 30, 'k', 'filled', 'MarkerFaceAlpha', 0.2);
plot(x_true_history(1,1), x_true_history(2,1), '.g', 'MarkerSize', 30, 'LineWidth', 3);
plot(x_true_history(1,end), x_true_history(2,end), '.r', 'MarkerSize', 30, 'LineWidth', 3);
xlabel('X Position');
ylabel('Y Position');
title('2D Tracking Results');
legend('True Path', 'Estimated Path', 'Final Particles', 'Start', 'End', 'Location', 'best');
grid on;
axis equal;

% X Position over time
subplot(2,3,2);
time = 0:T;
plot(time, x_true_history(1,:), 'b-', 'LineWidth', 2);
hold on;
plot(time, x_est_history(1,:), 'r--', 'LineWidth', 2);
xlabel('Time Step');
ylabel('X Position');
title('X Coordinate Tracking');
legend('True', 'Estimated');
grid on;

% Y Position over time
subplot(2,3,3);
plot(time, x_true_history(2,:), 'b-', 'LineWidth', 2);
hold on;
plot(time, x_est_history(2,:), 'r--', 'LineWidth', 2);
xlabel('Time Step');
ylabel('Y Position');
title('Y Coordinate Tracking');
legend('True', 'Estimated');
grid on;

% Effective Sample Size
subplot(2,3,4);
plot(1:T, Neff_history, 'b-', 'LineWidth', 2);
hold on;
plot([1 T], [threshold threshold], 'r--', 'LineWidth', 2);
xlabel('Time Step');
ylabel('Neff');
title('Effective Sample Size History');
legend('Neff', sprintf('Threshold (%.1f)', threshold), 'Location', 'best');
grid on;
ylim([0 N+1]);

% Position Errors
subplot(2,3,5);
x_error = x_true_history(1,:) - x_est_history(1,:);
y_error = x_true_history(2,:) - x_est_history(2,:);
position_error = sqrt(x_error.^2 + y_error.^2);
plot(time, position_error, 'k-', 'LineWidth', 2);
hold on;
plot([0 T], [mean(position_error) mean(position_error)], 'r--', 'LineWidth', 1.5);
xlabel('Time Step');
ylabel('Position Error');
title(sprintf('2D Position Error\nMean = %.4f, RMS = %.4f', ...
      mean(position_error), sqrt(mean(position_error.^2))));
legend('Error', sprintf('Mean (%.4f)', mean(position_error)));
grid on;

% Resampling 
subplot(2,3,6);
resample_percentage = sum(resample_flags) / T * 100;
bar([1 2], [sum(resample_flags), T-sum(resample_flags)]);
set(gca, 'XTickLabel', {'Resampled', 'Not Resampled'});
ylabel('Number of Time Steps');
title(sprintf('Resampling Frequency: %.1f%%', resample_percentage));
grid on;

