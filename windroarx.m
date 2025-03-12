clear; clc; close all;
yalmip('clear');

rng(42);

% 参数初始化
n = 2;          % 资源数量
K = 40;         % 总数据点数量
train_size = 30; % 训练集大小
p = 0.8;        % 覆盖率参数
M = 1e6;        % 大数
n_a = 1;        % 输出滞后阶数
n_b = 1;        % 输入滞后阶数

% 生成真实回归系数
true_a = rand(n_a, n);          % 输出滞后项系数 (n_a x n 矩阵)
true_b = rand(n_b, n);          % 输入滞后项系数 (n_b x n 矩阵)

% 生成预测模型输入
x = rand(K, 1);                 % 输入矩阵 (K x 1)

% 根据 ARX 模型生成输出，带有噪声
y = zeros(K, n);
noise_level = 0.5; % 噪声水平

for k = 1:K
    if k > max(n_a, n_b)
        for i = 1:n
            y(k, i) = sum(true_a(:, i) .* flip(y(k-n_a:k-1, i))) + ...
                      sum(true_b(:, i) .* flip(x(k-n_b:k-1, 1))) + ...
                      noise_level * randn;
        end
    else
        % 前几步数据不足时，仅添加噪声
        y(k, :) = noise_level * randn(1, n);
    end
end

% 分割训练和测试数据
x_train = x(1:train_size);
y_train = y(1:train_size, :);

% 定义YALMIP变量
beta_a = sdpvar(n_a, n);        % 输出滞后项系数 (n_a x n)
beta_b = sdpvar(n_b, n);        % 输入滞后项系数 (n_b x n)
delta = sdpvar(n, 1);           % 不确定性集大小
z = binvar(train_size, n);      % 二进制变量 (train_size x n)
epsilon = sdpvar(train_size, n); % 绝对值辅助变量 (train_size x n)

% 定义目标函数
objective = (1/n) * sum(sum(epsilon)) %+ sum(delta);

% 约束条件初始化
constraints = [];
for k = 1:train_size
    if k > max(n_a, n_b)
        for i = 1:n
            % 预测值
            y_pred = sum(beta_a(:, i) .* flip(y_train(k-n_a:k-1, i))) + ...
                     sum(beta_b(:, i) .* flip(x_train(k-n_b:k-1, 1)));
                 
            % 引入绝对值约束
            constraints = [constraints, y_train(k, i) - y_pred <= epsilon(k, i)];
            constraints = [constraints, y_pred - y_train(k, i) <= epsilon(k, i)];

            % 不确定性集约束
            constraints = [constraints, epsilon(k, i) <= delta(i) + M * (1 - z(k, i))];
        end
    end
end

for i = 1:n
    constraints = [constraints, sum(z(:, i)) >= p * train_size];  % 覆盖率约束
    constraints = [constraints, delta(i) >= 0];                   % 不确定性非负
    constraints = [constraints, epsilon(:, i) >= 0];              % 绝对值非负
    constraints = [constraints, beta_a(:, i) >= 0];
    constraints = [constraints, beta_b(:, i) >= 0];               % 回归系数非负
end

% 配置求解器
options = sdpsettings('solver', 'scip', 'verbose', 1);

% 求解优化问题
sol = optimize(constraints, objective, options);

% 检查求解状态
if sol.problem == 0
    disp('优化成功！');
    disp('输出滞后项系数 beta_a:');
    disp(value(beta_a));
    disp('输入滞后项系数 beta_b:');
    disp(value(beta_b));
    disp('不确定性集大小 delta:');
    disp(value(delta));
else
    disp('优化失败！');
    disp(yalmiperror(sol.problem));
end

% 全部数据预测
y_fit = zeros(K, n);
for k = 1:K
    if k > max(n_a, n_b)
        for i = 1:n
            y_fit(k, i) = sum(value(beta_a(:, i)) .* flip(y(k-n_a:k-1, i))) + ...
                          sum(value(beta_b(:, i)) .* flip(x(k-n_b:k-1, 1)));
        end
    end
end

% 可视化结果
figure;
hold on;
for i = 1:n
    subplot(n, 1, i);
    hold on;
    % 实际值
    plot(1:K, y(:, i), 'ro-', 'DisplayName', ['实际值 (资源 ' num2str(i) ')']);
    % 预测值
    plot(1:K, y_fit(:, i), 'b*-', 'DisplayName', ['预测值 (资源 ' num2str(i) ')']);
    legend;
    xlabel('数据点');
    ylabel('风力容量');
    title(['资源 ' num2str(i) ' 实际值与预测值对比']);
    hold off;
end
hold off;
