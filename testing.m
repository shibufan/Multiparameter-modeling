clear;
clc;

%% 1.read data & models
load('Test_Data.mat') %read test data

load('trained moel/ANN/net.mat') %read ANN model
load('trained moel/SVM/svm.mat') %read SVM model
%% 2.Testing
%1.SVM predict
% actual order of TF in test data
test_responses_size = size(test_responses(:,1),1);
for i=1:test_responses_size
    freq=test_responses{i,1}(:,1);
    data=test_responses{i,1}(:,2)+test_responses{i,1}(:,3)*1j;
    fit_data=rationalfit(freq,data);
    test_order_TF(i)=size(fit_data.A(:,1),1);
end

% normalize test data
SVM_Test_matrix = mapminmax('apply',test_candidates',PS);
SVM_Test_matrix = SVM_Test_matrix';
SVM_Test_label = test_order_TF';

% predict orders of TF
[predict_SVM_test_label,accuracy_SVM_test,prob_estimates2] = svmpredict(SVM_Test_label,SVM_Test_matrix,svm_model);
SVM_test_result = [SVM_Test_label predict_SVM_test_label];

figure(1);
plot(SVM_Test_label,'r *');
hold on;
plot(predict_SVM_test_label,'b o');
grid on;
legend('Actual orders','SVM outputs');
xlabel('Testing samples');
ylabel('Order');
axis([0 40 5.5 12.5]);
string = {'SVM Predict result(RBF Kernel function) for Test Data ';
          ['accuracy = ' num2str(accuracy_SVM_test(1)) '%']};
title(string);

% 2. ANN prediction
test_size=size(test_responses(:,1),1);
test_data=cell(test_size,1);
test_outputs=cell(test_size,1);

for i=1:test_size
    test_data{i,1}(:,1)=test_responses{i,1}(:,1);
    test_outputs{i,1}(:,1:2)=test_responses{i,1}(:,2:3);
    for j=1:size(test_data{i,1}(:,1),1)
        test_data{i,1}(j,2:4)=test_candidates(i,1:3);
    end
end

predict_outputs=cell(test_size,1);
for i=1:test_size
    P_test=test_data{i,1}';
    if predict_SVM_test_label(i,1)==6
        p_test = mapminmax('apply',P_test,ps_input_1);
        t_sim = sim(net1,p_test);
        T_sim = mapminmax('reverse',t_sim,ps_output_1);
    end
    
    if predict_SVM_test_label(i,1)==7
        p_test = mapminmax('apply',P_test,ps_input_2);
        t_sim = sim(net2,p_test);
        T_sim = mapminmax('reverse',t_sim,ps_output_2);
    end
    
    if predict_SVM_test_label(i,1)==8
        p_test = mapminmax('apply',P_test,ps_input_3);
        t_sim = sim(net3,p_test);
        T_sim = mapminmax('reverse',t_sim,ps_output_3);
    end
    
    if predict_SVM_test_label(i,1)==10
        p_test = mapminmax('apply',P_test,ps_input_4);
        t_sim = sim(net4,p_test);
        T_sim = mapminmax('reverse',t_sim,ps_output_4);
    end
    T_sim=T_sim';
    predict_outputs{i,1}=T_sim;
end

N=0;
PE=0;
APE=0;
error_matrix=cell(test_size,1);
for i=1:test_size
    error_matrix{i,1}=abs((test_outputs{i,1}-predict_outputs{i,1})./test_outputs{i,1});
    complex_S11_test = test_outputs{i,1}(:,1)+test_outputs{i,1}(:,2)*1j;
    S11_test_abs=abs(complex_S11_test);
    complex_S11_predict = predict_outputs{i,1}(:,1)+predict_outputs{i,1}(:,2)*1j;
    S11_predict_abs=abs(complex_S11_predict);
    APE_matrix=abs((S11_test_abs-S11_predict_abs)./S11_test_abs);
    for j=1:size(test_outputs{i,1}(:,1),1)
        error_real=abs((test_outputs{i,1}(j,1)-predict_outputs{i,1}(j,1))/test_outputs{i,1}(j,1));
        error_imag=abs((test_outputs{i,1}(j,2)-predict_outputs{i,1}(j,2))/test_outputs{i,1}(j,2));
        PE=PE+sqrt(error_real^2+error_imag^2);
        APE=APE+APE_matrix(j);
        N=N+1;
    end
    
end
MPE=PE/N;
MAPE=APE/N;
disp(['MPE: ' num2str(MPE*100) '%'])
disp(['MAPE: ' num2str(MAPE*100) '%'])

for i=1:test_size
    wf = test_responses{i,1}(:,1);
    complex_S11_test = test_outputs{i,1}(:,1)+test_outputs{i,1}(:,2)*1j;
    S11_test_abs=abs(complex_S11_test);
    S11_test_dB = 20*log10(S11_test_abs);
    complex_S11_predict = predict_outputs{i,1}(:,1)+predict_outputs{i,1}(:,2)*1j;
    S11_predict_abs=abs(complex_S11_predict);
    S11_predict_dB = 20*log10(S11_predict_abs);
    figure(i+1);
    plot(wf,S11_test_dB,'r','LineWidth', 0.5);
    hold on;
    plot(wf,S11_predict_dB,'g','LineWidth', 0.5);
    legend('Actual responses','Predict responses')
    xlabel('Freq. in GHz')
    ylabel ('S_1_1 in dB')
    string = {['Multiparameter modeling predict result for testing sample ' num2str(i)];
        ['MAPE = ' num2str(MAPE*100) '%']};
    title(string);
    grid on
end