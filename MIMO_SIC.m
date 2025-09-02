clear;clc;close all;
rng('default')


%% 参数设置
NTx = 8;                % 发射天线数量
NRx = 16;                % 接收天线数量
SNR_dB = 10:2:20;        % 信噪比范围 (dB)
N_snr = length(SNR_dB); % SNR点数
N0 = 1./(10.^(SNR_dB/10));

M_mod = 16;
M_bits = log2(M_mod);

alphabet = qammod(0:M_mod-1,M_mod,"gray","InputType","integer","UnitAveragePower",true);

lld_k_Initial = log(1/M_mod * ones(1, M_mod));

frame = 1000;

%% 初始化误码率计算
avg_ber = zeros(1,N_snr);
avg_ber1 = zeros(1,N_snr);

%% 主仿真循环
for snr_idx = 1:N_snr
    currentSNR = SNR_dB(snr_idx);
    error_sum = 0;
    error_sum1 = 0;
    for ifram = 1:frame
        rng(ifram)

        % 生成随机数据
        tx_bits = randi([0 1], NTx*M_bits, 1);

        x = qammod(tx_bits,M_mod,"gray","InputType","bit","UnitAveragePower",true);


        % 通过 MIMO 信道传输
        % 生成瑞利衰落信道
        H = (randn(NRx, NTx) + 1i*randn(NRx, NTx)) / sqrt(NTx);

        y = H*x;

        % 添加高斯白噪声
        y = awgn(y, currentSNR, 'measured');

        X_Initial = (H'*H+N0(snr_idx))^(-1)*(H')*y;
        constellation = alphabet.';
        var_Initial = zeros(NTx,1);
        for k = 1:NTx
            var_Initial(k) = sum(abs(constellation - X_Initial(k)).^2 .* exp(lld_k_Initial.') / sum(exp(lld_k_Initial.')));
        end

        x_hat= iterative_SIC(X_Initial,var_Initial, N0(snr_idx), y, H, alphabet.', 10);

        rx_bits = qamdemod(x_hat,M_mod,"gray","OutputType","bit","UnitAveragePower",true);

        rx_bits1 = qamdemod(X_Initial,M_mod,"gray","OutputType","bit","UnitAveragePower",true);

        error_sum = error_sum+sum(xor(rx_bits,tx_bits));

        avg_ber(1,snr_idx) = error_sum./ifram/(NTx*M_bits);

        error_sum1 = error_sum1+sum(xor(rx_bits1,tx_bits));

        avg_ber1(1,snr_idx) = error_sum1./ifram/(NTx*M_bits);


        clc
        display(ifram,'Number of frames');
        display(avg_ber,'Average BER - iterative-SIC');
        display(avg_ber1,'Average BER - MMSE');
    end
end

%% 绘制结果
figure;
semilogy(SNR_dB, avg_ber, 'r-o', 'LineWidth', 2);
hold on
semilogy(SNR_dB, avg_ber1, 'b-x', 'LineWidth', 2);
grid on;
xlabel('SNR (dB)');
ylabel('Bit Error Rate (BER)');
legend('MIMO with iterative-SIC detection','MIMO with LMMSE detection');

title_text = ['NtxNr = ',num2str(NTx),'x',num2str(NRx)];
title(title_text);

function x_hat= iterative_SIC(x_Initial,var_Initial, N0, y, H, constellation, num_iter)

[Nr,Nt] = size(H);

K = length(constellation);

x_hat = x_Initial;
var_hat = var_Initial;
z = zeros(Nr,Nt);
sigma2 = zeros(Nr,Nt);
Le = zeros(Nr,Nt,K);

beta = 0.8;%阻尼因子，在采用高阶调制时是非常需要的

for iter = 1:num_iter

    for nr = 1:Nr
        hx = sum((H(nr,:).').*x_hat);
        hx_var = sum((abs(H(nr,:)).').^2.*var_hat);
        for nt = 1:Nt
            z(nr,nt) = y(nr)-hx+H(nr,nt)*x_hat(nt);
            sigma2(nr,nt) = N0+hx_var-(abs(H(nr,nt))^2)*var_hat(nt);
            for k = 1:K
                Le(nr,nt,k) = log(exp(-(abs(z(nr,nt)-H(nr,nt)*constellation(k))^2)/sigma2(nr,nt))/(pi*sigma2(nr,nt)));
            end
        end
    end

    Le_pp1 = squeeze(sum(Le,1));
    p = softmax(Le_pp1.').';

    x_hat = (1-beta)*x_hat + beta*(p*constellation);
    var_hat = (1-beta)*var_hat + beta*sum((abs(constellation.'-x_hat).^2).*p,2);

end

end






















